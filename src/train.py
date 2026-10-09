import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split
from tqdm import tqdm

from src.checkpoint import save_checkpoint
from src.config import ModelConfig, tiny_config
from src.data.dataset import MockMultimodalDataset
from src.data.loading import build_loader, load_manifest
from src.engine import evaluate, train_one_epoch
from src.models.backbones import TINY_IMAGE_SIZE
from src.models.multimodal_transformer import MultimodalDetector


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_config(args) -> ModelConfig:
    if args.tiny:
        return tiny_config()
    return ModelConfig(
        text_model=args.text_model,
        audio_model=args.audio_model,
        vision_model=args.vision_model,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        max_audio_seconds=args.max_audio_seconds,
    )


def build_datasets(args, cfg):
    if args.train_manifest:
        train_set = load_manifest(args.train_manifest, cfg)
    else:
        print("No --train_manifest given: training on synthetic mock data (smoke test only).")
        image_size = TINY_IMAGE_SIZE if cfg.tiny else 224
        train_set = MockMultimodalDataset(
            args.mock_size, image_size=image_size, audio_len=cfg.max_audio_len, sample_rate=cfg.sample_rate, seed=args.seed
        )
    if args.val_manifest:
        return train_set, load_manifest(args.val_manifest, cfg)
    n_val = int(len(train_set) * args.val_fraction)
    if n_val == 0:
        return train_set, None
    generator = torch.Generator().manual_seed(args.seed)
    return tuple(random_split(train_set, [len(train_set) - n_val, n_val], generator=generator))


def _fmt(value):
    return "n/a" if value is None else f"{value:.4f}"


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_config(args)
    train_set, val_set = build_datasets(args, cfg)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = build_loader(train_set, cfg, args.batch_size, True, args.num_workers, generator)
    val_loader = build_loader(val_set, cfg, args.batch_size, False, args.num_workers) if val_set else None
    print(f"Device: {device} | train: {len(train_set)} | val: {len(val_set) if val_set else 0}")

    model = MultimodalDetector(cfg).to(device)
    if args.freeze_backbones:
        model.freeze_backbones()
    groups = [
        {"params": model.backbone_parameters(), "lr": args.lr_backbone},
        {"params": model.head_parameters(), "lr": args.lr_head},
    ]
    optimizer = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=args.weight_decay)

    best_score, history = float("inf"), []
    for epoch in range(1, args.epochs + 1):
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        train_loss = train_one_epoch(
            model, progress, optimizer, device, args.adv_epsilon, args.adv_weight, args.grad_clip
        )
        record = {"epoch": epoch, "train_loss": train_loss}
        score = train_loss
        if val_loader:
            record["val"] = evaluate(model, val_loader, device, args.threshold)
            if record["val"]["loss"] is not None:
                score = record["val"]["loss"]
        history.append(record)

        summary = f"epoch {epoch}: train_loss={train_loss:.4f}"
        if "val" in record:
            val = record["val"]
            summary += f" val_loss={_fmt(val['loss'])}"
            for task in ("phishing", "deepfake"):
                summary += f" {task}_f1={_fmt(val[task]['f1'] if val[task] else None)}"
        print(summary)

        if score < best_score:
            best_score = score
            save_checkpoint(output_dir / "best.pt", model, epoch=epoch, metrics=record, threshold=args.threshold)
            print(f"  saved {output_dir / 'best.pt'}")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"Done. Best checkpoint: {output_dir / 'best.pt'}")
    return history


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the multimodal deepfake-phishing detector")
    data = p.add_argument_group("data")
    data.add_argument("--train_manifest", help="CSV manifest (see README). Omit to use synthetic mock data.")
    data.add_argument("--val_manifest", help="Separate validation manifest; otherwise --val_fraction is split off")
    data.add_argument("--val_fraction", type=float, default=0.2)
    data.add_argument("--mock_size", type=int, default=128)
    data.add_argument("--num_workers", type=int, default=0)

    model = p.add_argument_group("model")
    model.add_argument("--text_model", default="bert-base-uncased")
    model.add_argument("--audio_model", default="facebook/wav2vec2-base")
    model.add_argument("--vision_model", default="google/vit-base-patch16-224")
    model.add_argument("--d_model", type=int, default=512)
    model.add_argument("--n_heads", type=int, default=8)
    model.add_argument("--n_layers", type=int, default=4)
    model.add_argument("--dropout", type=float, default=0.1)
    model.add_argument("--max_audio_seconds", type=float, default=4.0)
    model.add_argument("--freeze_backbones", action="store_true", help="Train only the fusion and heads")
    model.add_argument("--tiny", action="store_true", help="Tiny random backbones, no download (smoke tests)")

    train = p.add_argument_group("training")
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch_size", type=int, default=8)
    train.add_argument("--lr_backbone", type=float, default=2e-5)
    train.add_argument("--lr_head", type=float, default=2e-4)
    train.add_argument("--weight_decay", type=float, default=1e-4)
    train.add_argument("--grad_clip", type=float, default=1.0)
    train.add_argument("--adv_epsilon", type=float, default=0.05, help="FGSM radius on modality features; 0 disables")
    train.add_argument("--adv_weight", type=float, default=0.3, help="Weight of the adversarial loss term")
    train.add_argument("--threshold", type=float, default=0.5)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--output_dir", default="outputs/run")
    return p.parse_args(argv)


if __name__ == "__main__":
    main()

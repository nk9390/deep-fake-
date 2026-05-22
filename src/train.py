import argparse
import sys
from pathlib import Path
import os

# Ensure project root is on sys.path when running this file directly (python src/train.py)
# This allows imports like `from src.models...` to resolve correctly.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoImageProcessor
from tqdm import tqdm
from src.models.multimodal_transformer import MultimodalDetector
from src.data.dataset import MockMultimodalDataset, ManifestImageTextDataset
from src.utils.adversarial import fgsm_on_features


def collate_fn(batch, tokenizer, image_processor, max_audio_len=32000):
    texts = [b["text"] for b in batch]
    tok = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")

    # Pad/trim audio to fixed length for simplicity
    audio_vals = [b["audio_values"][:max_audio_len] for b in batch]
    audio_vals = [torch.nn.functional.pad(v, (0, max(0, max_audio_len - v.size(0)))) for v in audio_vals]
    audio_values = torch.stack(audio_vals, dim=0)  # [B, T]

    images = [b["image"] for b in batch]
    img_inputs = image_processor(images=images, return_tensors="pt")
    pixel_values = img_inputs["pixel_values"]  # [B, 3, 224, 224]

    phishing = torch.tensor([b["phishing"] for b in batch], dtype=torch.long)
    deepfake = torch.tensor([b["deepfake"] for b in batch], dtype=torch.long)

    return {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "audio_values": audio_values,
        "pixel_values": pixel_values,
        "phishing": phishing,
        "deepfake": deepfake,
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    image_processor = AutoImageProcessor.from_pretrained(args.vision_model)

    if args.dataset_type == "manifest":
        dataset = ManifestImageTextDataset(
            manifest_path=args.manifest_path,
            image_root=args.image_root,
            image_size=224,
            audio_len=args.audio_len,
        )
    else:
        dataset = MockMultimodalDataset(length=args.dataset_size, image_size=224, audio_len=args.audio_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, tokenizer, image_processor, max_audio_len=args.audio_len),
    )

    model = MultimodalDetector(
        text_model_name=args.text_model,
        audio_model_name=args.audio_model,
        vision_model_name=args.vision_model,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        freeze_backbones=args.freeze_backbones,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            # Forward (clean)
            out = model(batch)
            loss_clean = ce(out["logits_phishing"], batch["phishing"]) + ce(out["logits_deepfake"], batch["deepfake"])

            # Prepare feature dict for adversarial attack
            features = {
                "text_feat": out["text_feat"],
                "audio_feat": out["audio_feat"],
                "vision_feat": out["vision_feat"],
            }
            # FGSM on features
            adv_feats = fgsm_on_features(
                model,
                features,
                targets={"phishing": batch["phishing"], "deepfake": batch["deepfake"]},
                epsilon=args.adv_epsilon,
            )
            out_adv = model.forward_from_features(adv_feats)
            loss_adv = ce(out_adv["logits_phishing"], batch["phishing"]) + ce(out_adv["logits_deepfake"], batch["deepfake"])

            # TRADES-style combination
            loss = args.alpha * loss_clean + (1.0 - args.alpha) * loss_adv

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "clean": f"{loss_clean.item():.4f}",
                "adv": f"{loss_adv.item():.4f}",
            })

    print("Training finished.")
    # Save checkpoint
    os.makedirs("outputs", exist_ok=True)
    ckpt_path = os.path.join("outputs", "model.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_model", type=str, default="bert-base-uncased")
    parser.add_argument("--audio_model", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--vision_model", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--dataset_size", type=int, default=512)
    parser.add_argument("--audio_len", type=int, default=32000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze_backbones", action="store_true")
    parser.add_argument("--adv_epsilon", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight for clean vs adversarial loss (TRADES)")
    parser.add_argument("--dataset_type", type=str, default="mock", choices=["mock", "manifest"], help="Choose dataset implementation")
    parser.add_argument("--manifest_path", type=str, default="data/test/manifest.txt", help="Path to image-text manifest file")
    parser.add_argument("--image_root", type=str, default="data/test", help="Directory containing images referenced by the manifest")
    args = parser.parse_args()
    train(args)
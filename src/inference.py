import argparse
import csv
from pathlib import Path

import torch

from src.checkpoint import load_checkpoint
from src.data.dataset import ManifestDataset
from src.data.loading import build_loader, load_manifest
from src.engine import predict
from src.utils.phishing_indicators import analyze_text

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
FIELDS = ["id", "phishing_prob", "phishing_pred", "deepfake_prob", "deepfake_pred", "red_flags"]


def build_dataset(args, cfg):
    kwargs = {"sample_rate": cfg.sample_rate, "max_audio_seconds": cfg.max_audio_seconds}
    if args.manifest:
        return load_manifest(args.manifest, cfg)
    if args.image_dir:
        images = sorted(p for p in Path(args.image_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not images:
            raise SystemExit(f"No images found in {args.image_dir}")
        return ManifestDataset([{"image": str(p)} for p in images], **kwargs)
    if args.text or args.audio or args.image:
        row = {"id": args.id, "text": args.text, "audio": args.audio, "image": args.image}
        return ManifestDataset([row], **kwargs)
    raise SystemExit("Give --manifest, --image_dir, or at least one of --text/--audio/--image")


def main(argv=None):
    p = argparse.ArgumentParser(description="Score messages/media for phishing and deepfakes")
    p.add_argument("--checkpoint", default="outputs/run/best.pt")
    p.add_argument("--manifest", help="CSV manifest (labels optional)")
    p.add_argument("--image_dir", help="Score every image in a directory")
    p.add_argument("--text", help="Single sample: message text")
    p.add_argument("--audio", help="Single sample: audio file")
    p.add_argument("--image", help="Single sample: image file")
    p.add_argument("--id", default="sample")
    p.add_argument("--output", default="outputs/predictions.csv")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args(argv)

    if not Path(args.checkpoint).is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint} (train one with: python -m src.train)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(args.checkpoint, device)
    loader = build_loader(build_dataset(args, model.cfg), model.cfg, args.batch_size)
    results = predict(model, loader, device)

    rows = []
    for i, sample_id in enumerate(results["ids"]):
        row = {"id": sample_id, "red_flags": ";".join(analyze_text(results["texts"][i])["indicators"])}
        for task in ("phishing", "deepfake"):
            prob = results["probs"][task][i]
            row[f"{task}_prob"] = round(prob, 4)
            row[f"{task}_pred"] = int(prob >= args.threshold)
        rows.append(row)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(f"{row['id']}: phishing={row['phishing_prob']:.3f} deepfake={row['deepfake_prob']:.3f} "
              f"red_flags=[{row['red_flags']}]")
    print(f"Wrote {len(rows)} predictions to {args.output}")
    return rows


if __name__ == "__main__":
    main()

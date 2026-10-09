import argparse
import json
from pathlib import Path

import torch

from src.checkpoint import load_checkpoint
from src.data.loading import build_loader, load_manifest
from src.engine import evaluate


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on a labelled manifest")
    p.add_argument("--checkpoint", default="outputs/run/best.pt")
    p.add_argument("--manifest", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output", help="Write metrics JSON here")
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(args.checkpoint, device)
    loader = build_loader(load_manifest(args.manifest, model.cfg), model.cfg, args.batch_size)
    metrics = evaluate(model, loader, device, args.threshold)
    print(json.dumps(metrics, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()

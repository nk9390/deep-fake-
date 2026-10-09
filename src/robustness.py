"""
Adversarial robustness report: how often can a white-box attacker make a malicious sample
(phishing = 1 or deepfake = 1) that the detector catches look benign, for growing budgets?
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.checkpoint import load_checkpoint
from src.data.loading import build_loader, load_manifest
from src.engine import to_device
from src.utils.adversarial import feature_attack, input_attack
from src.utils.losses import TASKS


def _probs(out):
    return {t: out[f"logits_{t}"].softmax(dim=-1)[:, 1].detach().cpu().numpy() for t in TASKS}


def attack_batch(model, batch, space, epsilon, steps):
    if space == "feature":
        with torch.no_grad():
            features = model.encode(batch)
        delta = feature_attack(model, features, batch, epsilon, steps=steps, random_start=steps > 1)
        with torch.no_grad():
            return model.classify({k: v + delta[k] for k, v in features.items()}, batch["modality_mask"])
    modalities = ["vision", "audio"] if space == "input" else [space]
    adv_inputs = input_attack(model, batch, modalities, epsilon, steps=steps, random_start=steps > 1)
    with torch.no_grad():
        return model({**batch, **adv_inputs})


def robustness_report(model, loader, device, space="feature", epsilons=(0.0, 0.01, 0.05, 0.1), steps=10, threshold=0.5):
    model.eval()
    rows = []
    for eps in epsilons:
        labels = {t: [] for t in TASKS}
        clean = {t: [] for t in TASKS}
        adv = {t: [] for t in TASKS}
        for batch in loader:
            batch = to_device(batch, device)
            with torch.no_grad():
                clean_probs = _probs(model(batch))
            adv_probs = clean_probs if eps == 0 else _probs(attack_batch(model, batch, space, eps, steps))
            for t in TASKS:
                labels[t].append(batch[t].cpu().numpy())
                clean[t].append(clean_probs[t])
                adv[t].append(adv_probs[t])
        row = {"space": space, "epsilon": eps, "steps": steps}
        for t in TASKS:
            y = np.concatenate(labels[t])
            keep = y >= 0
            y = y[keep]
            c = np.concatenate(clean[t])[keep] >= threshold
            a = np.concatenate(adv[t])[keep] >= threshold
            caught = (y == 1) & c
            row[t] = {
                "clean_accuracy": float((c == y).mean()) if len(y) else None,
                "adversarial_accuracy": float((a == y).mean()) if len(y) else None,
                # Of the malicious samples caught on clean input, how many does the attack sneak past?
                "evasion_rate": float((caught & ~a).sum() / caught.sum()) if caught.any() else None,
            }
        rows.append(row)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description="Adversarial robustness report for a checkpoint")
    p.add_argument("--checkpoint", default="outputs/run/best.pt")
    p.add_argument("--manifest", required=True, help="Labelled manifest")
    p.add_argument(
        "--space",
        choices=["feature", "vision", "audio", "input"],
        default="feature",
        help="Where to perturb: modality embeddings, image pixels, audio samples, or both raw inputs",
    )
    p.add_argument("--epsilons", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1])
    p.add_argument("--steps", type=int, default=10, help="PGD steps (1 = FGSM)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output", help="Write the report JSON here")
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(args.checkpoint, device)
    loader = build_loader(load_manifest(args.manifest, model.cfg), model.cfg, args.batch_size)
    rows = robustness_report(model, loader, device, args.space, args.epsilons, args.steps, args.threshold)

    def fmt(v):
        return "  n/a" if v is None else f"{v:5.2f}"

    print(f"{'eps':>6} | {'task':8} | clean_acc | adv_acc | evasion")
    for row in rows:
        for t in TASKS:
            r = row[t]
            print(f"{row['epsilon']:6.3f} | {t:8} | {fmt(r['clean_accuracy']):>9} | "
                  f"{fmt(r['adversarial_accuracy']):>7} | {fmt(r['evasion_rate']):>7}")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    main()

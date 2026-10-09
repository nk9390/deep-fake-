import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def visualize_predictions(csv_path="outputs/predictions.csv", output="outputs/plots/predictions.png", threshold=0.5):
    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit(f"No predictions in {csv_path}")

    x = np.arange(len(df))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.6), 5))
    ax.bar(x - width / 2, df["phishing_prob"], width, label="phishing")
    ax.bar(x + width / 2, df["deepfake_prob"], width, label="deepfake")
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=1, label=f"threshold {threshold}")
    ax.set_xticks(x, df["id"], rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("probability")
    ax.set_title("Detector scores per sample")
    ax.legend()
    fig.tight_layout()

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(f"Saved plot to {output}")
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot predictions written by ml.inference")
    p.add_argument("--csv", default="outputs/predictions.csv")
    p.add_argument("--output", default="outputs/plots/predictions.png")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()
    visualize_predictions(args.csv, args.output, args.threshold)

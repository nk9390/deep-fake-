import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_predictions(csv_path="outputs/predictions.csv", save_dir="outputs/plots"):
    if not os.path.exists(csv_path):
        print(f" CSV file not found: {csv_path}")
        return

    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    if df.empty:
        print("⚠️ No data in predictions CSV.")
        return

    print(f"✅ Loaded {len(df)} predictions.")
    print(df.head())

    # --- Phishing probabilities ---
    plt.figure(figsize=(10, 6))
    plt.bar(df["filename"], df["phishing_prob"]) 
    plt.title("Phishing Probability per Image")
    plt.xlabel("Image Filename")
    plt.ylabel("Phishing Probability")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    phishing_plot = os.path.join(save_dir, "phishing_probs.png")
    plt.savefig(phishing_plot)
    plt.close()

    # --- Deepfake probabilities ---
    plt.figure(figsize=(10, 6))
    plt.bar(df["filename"], df["deepfake_prob"]) 
    plt.title("Deepfake Probability per Image")
    plt.xlabel("Image Filename")
    plt.ylabel("Deepfake Probability")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    deepfake_plot = os.path.join(save_dir, "deepfake_probs.png")
    plt.savefig(deepfake_plot)
    plt.close()

    print(f"📊 Saved plots in: {save_dir}")
    print(f"   - {phishing_plot}")
    print(f"   - {deepfake_plot}")

    try:
        # Try opening automatically on macOS
        if os.name == "posix":
            os.system(f"open {save_dir}")
    except Exception as e:
        print(f"⚠️ Could not open automatically: {e}")

if __name__ == "__main__":
    visualize_predictions()

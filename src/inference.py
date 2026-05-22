import argparse
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor
from src.models.multimodal_transformer import MultimodalDetector


# --- Helper: Load text captions / manifest ---
def load_manifest(manifest_path):
    captions = {}
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 2:
                    captions[parts[0]] = parts[1]
    return captions


# --- Helper: Load or synthesize image if missing ---
def load_image_or_synthetic(path, size=224):
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except Exception:
        import numpy as np
        return Image.fromarray(
            np.uint8(np.clip(np.random.rand(size, size, 3) * 255, 0, 255))
        )


# --- Inference main routine ---
def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    image_processor = AutoImageProcessor.from_pretrained(args.vision_model)

    model = MultimodalDetector(
        text_model_name=args.text_model,
        audio_model_name=args.audio_model,
        vision_model_name=args.vision_model,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        freeze_backbones=True,
    ).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f" Loading checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
    else:
        print(" No checkpoint found — using untrained model weights.")

    model.eval()

    img_dir = Path(args.data_dir)
    # auto-detect manifest name
    manifest_candidates = [
        args.manifest_path,
        img_dir / "manifest.txt",
        img_dir / "test_texts.txt",
    ]
    manifest_path = next(
        (Path(p) for p in manifest_candidates if p and os.path.exists(p)), None
    )
    captions = load_manifest(str(manifest_path)) if manifest_path else {}
    if manifest_path:
        print(f" Loaded captions from: {manifest_path}")

    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    # Restrict to single image if provided
    if args.image_path:
        if not os.path.exists(args.image_path):
            print(f" Image not found: {args.image_path}")
            return
        img_dir = Path(os.path.dirname(args.image_path) or ".")
        single_name = os.path.basename(args.image_path)
        print(f"📸 Running single-image inference on: {args.image_path}")
        image_files = [single_name]

    if not image_files:
        print("No image files found. Add images to the data directory.")
        return

    results = []

    for filename in sorted(image_files):
        image_path = img_dir / filename
        image = load_image_or_synthetic(str(image_path), size=224)
        text = captions.get(filename, f"Image {filename}")

        with torch.no_grad():
            # Prepare multimodal inputs
            tok = tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors="pt")
            img_inputs = image_processor(images=[image], return_tensors="pt")
            audio_values = torch.zeros((1, args.audio_len))  # dummy audio for now

            batch = {
                "input_ids": tok["input_ids"].to(device),
                "attention_mask": tok["attention_mask"].to(device),
                "pixel_values": img_inputs["pixel_values"].to(device),
                "audio_values": audio_values.to(device),
                "phishing": torch.tensor([0], dtype=torch.long).to(device),
                "deepfake": torch.tensor([0], dtype=torch.long).to(device),
            }

            out = model(batch)
            probs_phish = F.softmax(out["logits_phishing"], dim=-1)[0].cpu().tolist()
            probs_deepfake = F.softmax(out["logits_deepfake"], dim=-1)[0].cpu().tolist()

            p_phish = probs_phish[1]
            p_deepfake = probs_deepfake[1]
            pred_phish = int(p_phish >= args.threshold)
            pred_deepfake = int(p_deepfake >= args.threshold)

        print(
            f" {filename} | Text: {text[:40]}... | "
            f"Phish: {p_phish:.3f} | Deepfake: {p_deepfake:.3f}"
        )

        results.append({
            "filename": filename,
            "text": text,
            "phishing_prob": round(p_phish, 4),
            "deepfake_prob": round(p_deepfake, 4),
            "phishing_pred": pred_phish,
            "deepfake_pred": pred_deepfake,
        })

    # --- Write CSV output ---
    os.makedirs(Path(args.output_path).parent, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        f.write("filename,phishing_prob,deepfake_prob,phishing_pred,deepfake_pred,text\n")
        for r in results:
            f.write(
                f"{r['filename']},{r['phishing_prob']},{r['deepfake_prob']},"
                f"{r['phishing_pred']},{r['deepfake_pred']},\"{r['text']}\"\n"
            )

    print(f"\n Inference complete. Results saved to {args.output_path}")
    print(f" {len(results)} files processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multimodal phishing/deepfake inference")
    parser.add_argument("--checkpoint", type=str, default="outputs/model.pth")
    parser.add_argument("--data_dir", type=str, default="data/test")
    parser.add_argument("--image_path", type=str, default=None, help="Run inference for a single image only")
    parser.add_argument("--output_path", type=str, default="outputs/predictions.csv")
    parser.add_argument("--text_model", type=str, default="bert-base-uncased")
    parser.add_argument("--audio_model", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--vision_model", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--audio_len", type=int, default=32000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--manifest_path", type=str, default=None, help="Optional path to manifest file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    args = parser.parse_args()
    run_inference(args)

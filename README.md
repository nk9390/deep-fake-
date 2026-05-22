# Adversarial Detection of Deepfake-Based Phishing (Multimodal Transformer)

This project scaffolds a multimodal detector for deepfake-based phishing using text (BERT), audio (Wav2Vec2), and vision (ViT) with a fusion transformer and adversarial training on modality features.

## Features
- Multimodal encoders: BERT, Wav2Vec2, ViT
- Fusion transformer over modality tokens
- Multi-task heads: phishing classification and deepfake detection
- Adversarial training via FGSM on feature space (TRADES-style mixing)
- Mock dataset for end-to-end sanity checks

## Setup
```bash
python -m pip install -r requirements.txt
```

## Run (sanity check)
```bash
python src/train.py --epochs 1 --dataset_size 64 --batch_size 4
```

Training now also saves a checkpoint to `outputs/model.pth`.

## Inference (image + caption)
Use the simple inference script that reads images from a directory and optional captions from a manifest.

```bash
# Activate env
source .venv/bin/activate

# Run training once (saves outputs/model.pth)
python -m src.train --epochs 1 --dataset_size 16 --batch_size 2

# Prepare a manifest (optional)
echo "img1.jpg|A cat sitting on a sofa" > data/test/manifest.txt
echo "img2.jpg|A man riding a horse" >> data/test/manifest.txt

# Place images in data/test/ (or the script will use synthetic images)

# Run inference
python -m src.inference --checkpoint outputs/model.pth --data_dir data/test --output_path outputs/predictions.txt
```

Notes:
- Inference uses BERT/Vit tokenizers and processors under the hood and creates dummy audio to satisfy the model’s multimodal inputs.
- If no images are present, the script generates synthetic images so you can test end-to-end.

## Plug In Real Data
- Replace `MockMultimodalDataset` with a dataset that loads:
  - Text: email/DM content strings.
  - Audio: 16kHz mono waveforms.
  - Vision: frames (start with one frame per sample using ViT).
- Update the collate function to use `Wav2Vec2Processor` for robust audio padding/masking.

## Notes
- Hugging Face models will download on first run; ensure network access and disk space.
- Run from the project root or use absolute paths if needed.
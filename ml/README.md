# Optional: deepfake ML extension

Not needed for the cyber toolkit. This is a multimodal model (BERT for text, Wav2Vec2 for
audio, ViT for images, fused by a transformer) that scores a message for phishing and its
voice note or image for deepfakes, with adversarial-robustness testing.

It needs PyTorch and a GPU to train for real, and labelled data: FaceForensics++ or Celeb-DF
(fake faces), ASVspoof 2019 LA (fake voices), a phishing email corpus. Until trained on those,
its scores mean nothing.

```bash
pip install -r ml/requirements.txt
python -m ml.train --tiny --epochs 2              # smoke test, no downloads
python -m ml.train --train_manifest data/train.csv --val_manifest data/val.csv
python -m ml.evaluate   --checkpoint outputs/run/best.pt --manifest data/test.csv
python -m ml.robustness --checkpoint outputs/run/best.pt --manifest data/test.csv --space vision
python -m ml.inference  --checkpoint outputs/run/best.pt --manifest data/test/manifest.csv
```

Manifests are CSV files with columns `id,text,audio,image,phishing,deepfake`; paths are
relative to the manifest and labels are `1`, `0` or empty. See `data/test/manifest.csv`.

Security notes: checkpoints load with `torch.load(weights_only=True)` so a malicious `.pt`
file can't run code; inputs are validated and size-capped. `ml.robustness` measures the
**evasion rate**: how often a small, invisible perturbation flips a caught deepfake to "real".

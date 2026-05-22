import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class MockMultimodalDataset(Dataset):
    """
    Mock dataset producing synthetic multimodal inputs and labels.
    Replace this with a real dataset that returns:
      - text: str
      - audio_values: FloatTensor [T] at 16kHz
      - pixel_values: images or frames
      - labels: phishing (0/1), deepfake (0/1)
    """

    def __init__(self, length=256, image_size=224, audio_len=32000):
        self.length = length
        self.image_size = image_size
        self.audio_len = audio_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Synthetic text
        text = "Important update to your account details. Please verify immediately."
        # Synthetic audio waveform (2 seconds at 16kHz)
        audio_values = torch.randn(self.audio_len)
        # Synthetic RGB image (placeholder for a video frame or averaged frames)
        img = Image.fromarray(
            np.uint8(np.clip(np.random.rand(self.image_size, self.image_size, 3) * 255, 0, 255))
        )
        # Labels
        phishing = np.random.randint(0, 2)
        deepfake = np.random.randint(0, 2)
        return {"text": text, "audio_values": audio_values, "image": img, "phishing": phishing, "deepfake": deepfake}


class ManifestImageTextDataset(Dataset):
    """
    Minimal dataset that reads a manifest file with lines of the form:
      filename.jpg|caption text

    It loads the image from `image_root/filename.jpg` and uses the caption as text.
    Audio is synthesized as zeros for compatibility with the model.
    Labels default to 0 (benign/authentic) unless provided as optional fields:
      filename.jpg|caption text|phishing|deepfake
    """

    def __init__(self, manifest_path: str, image_root: str, image_size: int = 224, audio_len: int = 32000):
        self.manifest_path = manifest_path
        self.image_root = image_root
        self.image_size = image_size
        self.audio_len = audio_len
        self.items = []

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 2:
                    continue
                entry = {
                    "filename": parts[0],
                    "caption": parts[1],
                    "phishing": int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0,
                    "deepfake": int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0,
                }
                self.items.append(entry)

    def __len__(self):
        return len(self.items)

    def _load_image_or_synthetic(self, path):
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
        except Exception:
            # Fallback synthetic image if file missing or unreadable
            img = Image.fromarray(
                np.uint8(np.clip(np.random.rand(self.image_size, self.image_size, 3) * 255, 0, 255))
            )
        return img

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = os.path.join(self.image_root, item["filename"])
        image = self._load_image_or_synthetic(img_path)

        text = item["caption"]
        # Synthetic audio (zeros) to keep collate_fn simple
        audio_values = torch.zeros(self.audio_len)

        phishing = item["phishing"]
        deepfake = item["deepfake"]

        return {"text": text, "audio_values": audio_values, "image": image, "phishing": phishing, "deepfake": deepfake}
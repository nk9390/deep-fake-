"""
Datasets. Every sample is a dict:
  id: str, text: str|None, audio: FloatTensor [T] at `sample_rate`|None, image: PIL.Image|None,
  phishing: int, deepfake: int   (1 = malicious/fake, 0 = benign/authentic, -1 = unlabelled)
"""
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from PIL import Image
from torch.utils.data import Dataset

COLUMNS = ("id", "text", "audio", "image", "phishing", "deepfake")
# Inputs are attacker-controlled: cap what we read so one sample can't exhaust memory.
MAX_TEXT_CHARS = 20_000


def parse_label(value, column, row_number):
    value = (value or "").strip()
    if value in ("", "-1"):
        return -1
    if value in ("0", "1"):
        return int(value)
    raise ValueError(f"row {row_number}: {column} must be 0, 1 or empty, got {value!r}")


def load_audio(path, sample_rate, max_seconds):
    """Read at most `max_seconds` of a mono-mixed clip, resampled to `sample_rate`."""
    file_rate = sf.info(str(path)).samplerate
    data, file_rate = sf.read(str(path), frames=int(max_seconds * file_rate), dtype="float32", always_2d=True)
    audio = torch.from_numpy(data.mean(axis=1))
    if file_rate != sample_rate:
        audio = AF.resample(audio, file_rate, sample_rate)
    return audio


def load_image(path):
    # PIL raises DecompressionBombError for images far above Image.MAX_IMAGE_PIXELS.
    with Image.open(path) as img:
        return img.convert("RGB")


class ManifestDataset(Dataset):
    """
    Samples described by rows with the columns in COLUMNS (all optional, but each row needs at
    least one of text/audio/image). File paths are relative to `root`. Missing files and bad
    labels raise at construction time instead of silently producing garbage predictions.
    """

    def __init__(self, rows, root=".", sample_rate=16000, max_audio_seconds=4.0):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.max_audio_seconds = max_audio_seconds
        self.rows = [self._parse(row, i + 1) for i, row in enumerate(rows)]

    @classmethod
    def from_csv(cls, path, root=None, **kwargs):
        path = Path(path)
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            unknown = set(reader.fieldnames or []) - set(COLUMNS)
            if unknown:
                raise ValueError(f"{path}: unknown columns {sorted(unknown)}; expected {list(COLUMNS)}")
            rows = list(reader)
        return cls(rows, root=root or path.parent, **kwargs)

    def _resolve(self, value, row_number):
        value = (value or "").strip()
        if not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = self.root / path
        if not path.is_file():
            raise FileNotFoundError(f"row {row_number}: {path} not found")
        return path

    def _parse(self, row, row_number):
        text = (row.get("text") or "").strip()[:MAX_TEXT_CHARS] or None
        audio = self._resolve(row.get("audio"), row_number)
        image = self._resolve(row.get("image"), row_number)
        if not (text or audio or image):
            raise ValueError(f"row {row_number}: needs at least one of text, audio, image")
        sample_id = (row.get("id") or "").strip() or (image or audio or Path(f"row{row_number}")).name
        return {
            "id": sample_id,
            "text": text,
            "audio": audio,
            "image": image,
            "phishing": parse_label(row.get("phishing"), "phishing", row_number),
            "deepfake": parse_label(row.get("deepfake"), "deepfake", row_number),
        }

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {
            **row,
            "audio": load_audio(row["audio"], self.sample_rate, self.max_audio_seconds) if row["audio"] else None,
            "image": load_image(row["image"]) if row["image"] else None,
        }


PHISHING_TEXTS = [
    "URGENT: your account will be suspended. Verify your password at http://192.168.4.1/login now.",
    "This is your CEO. I need you to buy gift cards immediately and send me the codes.",
    "Security alert: unusual sign-in. Confirm your bank details within 24 hours at bit.ly/secure-acct.",
    "Your parcel is on hold. Pay the customs fee at http://xn--pypal-4ve.com/pay to release it.",
]
BENIGN_TEXTS = [
    "Hi team, the meeting notes from Tuesday are attached. See you at the standup.",
    "Thanks for the quick review, I merged the fix this morning.",
    "Reminder: the library closes early on Friday for maintenance.",
    "Can you send me the slides from yesterday's lecture when you get a chance?",
]


class MockMultimodalDataset(Dataset):
    """
    Synthetic, deterministic data with a learnable signal, for smoke tests only:
    phishing samples use phishing-style text; deepfake samples carry a checkerboard image
    artifact (like GAN upsampling) and a pure synthetic tone in the audio.
    """

    def __init__(self, length=256, image_size=224, audio_len=32000, sample_rate=16000, seed=0):
        self.length = length
        self.image_size = image_size
        self.audio_len = audio_len
        self.sample_rate = sample_rate
        self.seed = seed

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        phishing = int(rng.integers(0, 2))
        deepfake = int(rng.integers(0, 2))
        texts = PHISHING_TEXTS if phishing else BENIGN_TEXTS
        text = texts[int(rng.integers(0, len(texts)))]

        s = self.image_size
        gradient = np.linspace(0, 200, s)[None, :, None] * np.ones((s, 1, 3))
        img = gradient + rng.normal(0, 10, (s, s, 3))
        if deepfake:
            checker = (np.indices((s, s)).sum(axis=0) % 2)[..., None] * 60.0
            img = img + checker
        image = Image.fromarray(np.uint8(np.clip(img, 0, 255)))

        audio = rng.normal(0, 0.1, self.audio_len).astype(np.float32)
        if deepfake:
            t = np.arange(self.audio_len) / self.sample_rate
            audio = audio + 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        return {
            "id": f"mock{idx}",
            "text": text,
            "audio": torch.from_numpy(audio),
            "image": image,
            "phishing": phishing,
            "deepfake": deepfake,
        }

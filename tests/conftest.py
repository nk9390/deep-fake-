import csv

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from src.config import tiny_config


@pytest.fixture
def cfg():
    return tiny_config()


@pytest.fixture
def sample_dir(tmp_path):
    """A small labelled manifest with a full row, a text-only row and an unlabelled image row."""
    Image.fromarray(np.uint8(np.random.default_rng(0).integers(0, 255, (40, 48, 3)))).save(tmp_path / "frame.png")
    t = np.arange(1600) / 8000  # 0.2 s at 8 kHz, so loading must resample to 16 kHz
    sf.write(tmp_path / "voice.wav", (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32), 8000)
    rows = [
        {"id": "full", "text": "URGENT verify your password", "audio": "voice.wav", "image": "frame.png",
         "phishing": "1", "deepfake": "1"},
        {"id": "", "text": "See you at the standup", "audio": "", "image": "", "phishing": "0", "deepfake": "0"},
        {"id": "", "text": "", "audio": "", "image": "frame.png", "phishing": "", "deepfake": "1"},
    ]
    with open(tmp_path / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path

import torch
from PIL import Image

from ml.utils.losses import TASKS


class MultimodalCollator:
    """Batches samples into model inputs; absent modalities get placeholders and a False mask."""

    def __init__(self, tokenizer, image_processor, max_text_len=256, max_audio_len=64000):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_text_len = max_text_len
        self.max_audio_len = max_audio_len

    def __call__(self, samples):
        texts = [s["text"] or "" for s in samples]
        tok = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_text_len, return_tensors="pt")

        # Fixed-length, zero-padded, per-clip normalised audio (as Wav2Vec2FeatureExtractor does).
        audio_values = torch.zeros(len(samples), self.max_audio_len)
        audio_lengths = torch.zeros(len(samples), dtype=torch.long)
        for i, s in enumerate(samples):
            clip = s["audio"]
            if clip is None or clip.numel() == 0:
                continue
            clip = clip[: self.max_audio_len].float()
            clip = (clip - clip.mean()) / torch.sqrt(clip.var(unbiased=False) + 1e-7)
            audio_values[i, : clip.numel()] = clip
            audio_lengths[i] = clip.numel()

        images = [s["image"] if s["image"] is not None else Image.new("RGB", (224, 224)) for s in samples]
        pixel_values = self.image_processor(images=images, return_tensors="pt")["pixel_values"]

        modality_mask = torch.tensor(
            [[bool(s["text"]), s["audio"] is not None, s["image"] is not None] for s in samples], dtype=torch.bool
        )
        batch = {
            "ids": [s["id"] for s in samples],
            "texts": texts,
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "audio_values": audio_values,
            "audio_lengths": audio_lengths,
            "pixel_values": pixel_values,
            "modality_mask": modality_mask,
        }
        for task in TASKS:
            batch[task] = torch.tensor([s[task] for s in samples], dtype=torch.long)
        return batch

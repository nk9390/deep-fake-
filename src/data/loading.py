from torch.utils.data import DataLoader

from src.config import ModelConfig
from src.data.collate import MultimodalCollator
from src.data.dataset import ManifestDataset
from src.models.backbones import build_processors


def load_manifest(path, cfg: ModelConfig):
    return ManifestDataset.from_csv(path, sample_rate=cfg.sample_rate, max_audio_seconds=cfg.max_audio_seconds)


def build_loader(dataset, cfg: ModelConfig, batch_size=8, shuffle=False, num_workers=0, generator=None):
    tokenizer, image_processor = build_processors(cfg)
    collator = MultimodalCollator(tokenizer, image_processor, cfg.max_text_len, cfg.max_audio_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        generator=generator,
    )

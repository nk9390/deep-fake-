"""Backbone encoders (text, audio, vision) and the processors that feed them."""
import json

import torch
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    BertConfig,
    ViTConfig,
    ViTImageProcessor,
    Wav2Vec2Config,
)

from src.config import ModelConfig

MODALITIES = ("text", "audio", "vision")
TINY_IMAGE_SIZE = 32


class CharTokenizer:
    """Byte-level tokenizer used with the tiny backbones, so tests need no download."""

    pad_token_id, cls_token_id, sep_token_id = 0, 1, 2
    vocab_size = 256 + 3

    def __call__(self, texts, padding=True, truncation=True, max_length=512, return_tensors="pt"):
        seqs = [
            [self.cls_token_id] + [b + 3 for b in t.encode("utf-8")][: max_length - 2] + [self.sep_token_id]
            for t in texts
        ]
        width = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s)
            attention_mask[i, : len(s)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def tiny_backbone_configs() -> dict:
    return {
        "text": BertConfig(
            vocab_size=CharTokenizer.vocab_size,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=64,
        ),
        "audio": Wav2Vec2Config(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            conv_dim=(16, 16),
            conv_stride=(5, 4),
            conv_kernel=(10, 8),
            num_conv_pos_embeddings=16,
            num_conv_pos_embedding_groups=2,
            apply_spec_augment=False,
        ),
        "vision": ViTConfig(
            image_size=TINY_IMAGE_SIZE,
            patch_size=8,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
        ),
    }


def config_to_json(config) -> str:
    return config.to_json_string(use_diff=False)


def config_from_json(text: str):
    data = json.loads(text)
    return AutoConfig.for_model(data.pop("model_type"), **data)


def _build(name, pretrained, config=None, **kwargs):
    try:
        if config is not None:
            return AutoModel.from_config(config, **kwargs)
        if pretrained:
            return AutoModel.from_pretrained(name, **kwargs)
        return AutoModel.from_config(AutoConfig.from_pretrained(name), **kwargs)
    except TypeError:
        # Some architectures don't take add_pooling_layer.
        if not kwargs:
            raise
        return _build(name, pretrained, config)


def build_encoders(cfg: ModelConfig, configs: dict = None):
    """Return (text, audio, vision) encoders. `configs` rebuilds them offline from saved configs."""
    if configs is None and cfg.tiny:
        configs = tiny_backbone_configs()
    configs = configs or {}
    names = {"text": cfg.text_model, "audio": cfg.audio_model, "vision": cfg.vision_model}
    # The [CLS] hidden state is used directly, so the (randomly initialised) poolers are dropped.
    pooling = {"text": {"add_pooling_layer": False}, "audio": {}, "vision": {"add_pooling_layer": False}}
    return tuple(_build(names[m], cfg.pretrained, configs.get(m), **pooling[m]) for m in MODALITIES)


def build_processors(cfg: ModelConfig):
    """Return (tokenizer, image_processor) for the configured backbones."""
    if cfg.tiny:
        return CharTokenizer(), ViTImageProcessor(size={"height": TINY_IMAGE_SIZE, "width": TINY_IMAGE_SIZE})
    return AutoTokenizer.from_pretrained(cfg.text_model), AutoImageProcessor.from_pretrained(cfg.vision_model)

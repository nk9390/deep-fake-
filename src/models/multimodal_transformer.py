import torch
import torch.nn as nn

from src.config import ModelConfig
from src.models.backbones import MODALITIES, build_encoders, config_from_json, config_to_json


class MultimodalDetector(nn.Module):
    """
    Encodes text (BERT), audio (Wav2Vec2) and an image (ViT) into one token each, fuses the
    tokens with a transformer behind a learned [CLS] token, and predicts two tasks from it:
    phishing (0/1) and deepfake (0/1). Missing modalities are masked out of the fusion.
    """

    def __init__(self, cfg: ModelConfig, backbone_configs: dict = None):
        super().__init__()
        self.cfg = cfg
        if backbone_configs is not None:
            backbone_configs = {k: config_from_json(v) for k, v in backbone_configs.items()}
        self.text_encoder, self.audio_encoder, self.vision_encoder = build_encoders(cfg, backbone_configs)

        d = cfg.d_model
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, d)
        self.audio_proj = nn.Linear(self.audio_encoder.config.hidden_size, d)
        self.vision_proj = nn.Linear(self.vision_encoder.config.hidden_size, d)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls_token, std=0.02)
        self.modality_embeddings = nn.Embedding(len(MODALITIES), d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=d * 4, dropout=cfg.dropout, batch_first=True, norm_first=True
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

        self.head_phishing = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(d, 2))
        self.head_deepfake = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(d, 2))

    # --- parameter groups -------------------------------------------------
    def encoders(self):
        return [self.text_encoder, self.audio_encoder, self.vision_encoder]

    def freeze_backbones(self):
        for enc in self.encoders():
            enc.requires_grad_(False)

    def backbone_parameters(self):
        return [p for enc in self.encoders() for p in enc.parameters() if p.requires_grad]

    def head_parameters(self):
        backbone = {id(p) for enc in self.encoders() for p in enc.parameters()}
        return [p for p in self.parameters() if id(p) not in backbone and p.requires_grad]

    def backbone_configs(self) -> dict:
        """JSON configs of the backbones, so a checkpoint can be rebuilt without downloads."""
        return {m: config_to_json(enc.config) for m, enc in zip(MODALITIES, self.encoders())}

    # --- encoders ---------------------------------------------------------
    def encode_text(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.text_proj(out.last_hidden_state[:, 0])

    def encode_audio(self, audio_values, audio_lengths):
        enc = self.audio_encoder
        kwargs = {}
        # Models with layer-norm feature extractors expect an attention mask; group-norm ones
        # (e.g. wav2vec2-base) must not get one and are fed zero padding instead.
        if getattr(enc.config, "feat_extract_norm", None) == "layer":
            positions = torch.arange(audio_values.size(1), device=audio_values.device)
            kwargs["attention_mask"] = (positions[None] < audio_lengths.clamp(min=1)[:, None]).long()
        hidden = enc(input_values=audio_values, **kwargs).last_hidden_state
        if not hasattr(enc, "_get_feat_extract_output_lengths"):
            return self.audio_proj(hidden.mean(dim=1))
        # Mean-pool over the frames that came from real (non-padding) samples.
        frame_lengths = enc._get_feat_extract_output_lengths(audio_lengths).clamp(min=0, max=hidden.size(1))
        positions = torch.arange(hidden.size(1), device=hidden.device)
        mask = (positions[None] < frame_lengths[:, None]).unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.audio_proj(pooled)

    def encode_vision(self, pixel_values):
        out = self.vision_encoder(pixel_values=pixel_values)
        return self.vision_proj(out.last_hidden_state[:, 0])

    def encode(self, batch) -> dict:
        return {
            "text": self.encode_text(batch["input_ids"], batch["attention_mask"]),
            "audio": self.encode_audio(batch["audio_values"], batch["audio_lengths"]),
            "vision": self.encode_vision(batch["pixel_values"]),
        }

    # --- fusion + heads ---------------------------------------------------
    def classify(self, features: dict, modality_mask):
        """features: {modality: [B, d_model]}; modality_mask: [B, 3] bool, True where present."""
        x = torch.stack([features[m] for m in MODALITIES], dim=1) + self.modality_embeddings.weight[None]
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        cls_present = torch.ones_like(modality_mask[:, :1])
        padding_mask = ~torch.cat([cls_present, modality_mask], dim=1)
        rep = self.norm(self.fusion(x, src_key_padding_mask=padding_mask)[:, 0])
        return {"logits_phishing": self.head_phishing(rep), "logits_deepfake": self.head_deepfake(rep)}

    def forward(self, batch):
        features = self.encode(batch)
        out = self.classify(features, batch["modality_mask"])
        out["features"] = features
        return out

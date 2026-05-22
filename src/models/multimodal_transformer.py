import torch
import torch.nn as nn
from transformers import AutoModel


class MultimodalDetector(nn.Module):
    def __init__(
        self,
        text_model_name: str = "bert-base-uncased",
        audio_model_name: str = "facebook/wav2vec2-base",
        vision_model_name: str = "google/vit-base-patch16-224",
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        num_classes_phishing: int = 2,
        num_classes_deepfake: int = 2,
        freeze_backbones: bool = False,
    ):
        super().__init__()
        # Backbones
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.audio_encoder = AutoModel.from_pretrained(audio_model_name)
        self.vision_encoder = AutoModel.from_pretrained(vision_model_name)

        if freeze_backbones:
            for m in [self.text_encoder, self.audio_encoder, self.vision_encoder]:
                for p in m.parameters():
                    p.requires_grad = False

        # Project pooled outputs into a shared space
        hidden_sizes = {
            "text": self.text_encoder.config.hidden_size,
            "audio": self.audio_encoder.config.hidden_size,
            "vision": self.vision_encoder.config.hidden_size,
        }
        self.text_proj = nn.Linear(hidden_sizes["text"], d_model)
        self.audio_proj = nn.Linear(hidden_sizes["audio"], d_model)
        self.vision_proj = nn.Linear(hidden_sizes["vision"], d_model)

        # Modality type embeddings: [text, audio, vision]
        self.modality_type_embeddings = nn.Embedding(3, d_model)

        # Fusion transformer over modality tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Heads: multi-task classification
        self.classifier_phishing = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, num_classes_phishing)
        )
        self.classifier_deepfake = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, num_classes_deepfake)
        )

    def encode_text(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # Use [CLS] pooled output when available; fallback to mean pool
        pooled = (
            out.pooler_output
            if hasattr(out, "pooler_output") and out.pooler_output is not None
            else out.last_hidden_state[:, 0]
        )
        return self.text_proj(pooled)

    def encode_audio(self, audio_values, audio_attention_mask=None):
        out = self.audio_encoder(input_values=audio_values, attention_mask=audio_attention_mask, return_dict=True)
        # Mean pool over time
        pooled = out.last_hidden_state.mean(dim=1)
        return self.audio_proj(pooled)

    def encode_vision(self, pixel_values):
        out = self.vision_encoder(pixel_values=pixel_values, return_dict=True)
        # ViT provides pooled output
        pooled = out.pooler_output
        return self.vision_proj(pooled)

    def fuse(self, text_feat, audio_feat, vision_feat):
        # Stack modality tokens: shape [batch, 3, d_model]
        x = torch.stack([text_feat, audio_feat, vision_feat], dim=1)
        # Add modality type embeddings
        device = x.device
        modality_ids = torch.tensor([0, 1, 2], device=device).unsqueeze(0).repeat(x.size(0), 1)
        x = x + self.modality_type_embeddings(modality_ids)
        # Transformer fusion
        fused_tokens = self.fusion_transformer(x)
        # Use mean pooled fused representation
        fused_rep = fused_tokens.mean(dim=1)
        return fused_tokens, fused_rep

    def forward(self, batch):
        text_feat = self.encode_text(batch["input_ids"], batch["attention_mask"])
        audio_feat = self.encode_audio(batch["audio_values"], batch.get("audio_attention_mask", None))
        vision_feat = self.encode_vision(batch["pixel_values"])
        fused_tokens, fused_rep = self.fuse(text_feat, audio_feat, vision_feat)
        logits_phishing = self.classifier_phishing(fused_rep)
        logits_deepfake = self.classifier_deepfake(fused_rep)
        return {
            "logits_phishing": logits_phishing,
            "logits_deepfake": logits_deepfake,
            "text_feat": text_feat,
            "audio_feat": audio_feat,
            "vision_feat": vision_feat,
            "fused_tokens": fused_tokens,
        }

    def forward_from_features(self, features):
        fused_tokens, fused_rep = self.fuse(features["text_feat"], features["audio_feat"], features["vision_feat"])
        logits_phishing = self.classifier_phishing(fused_rep)
        logits_deepfake = self.classifier_deepfake(fused_rep)
        return {"logits_phishing": logits_phishing, "logits_deepfake": logits_deepfake, "fused_tokens": fused_tokens}
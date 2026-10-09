from dataclasses import asdict, dataclass, fields


@dataclass
class ModelConfig:
    text_model: str = "bert-base-uncased"
    audio_model: str = "facebook/wav2vec2-base"
    vision_model: str = "google/vit-base-patch16-224"
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    max_text_len: int = 256
    sample_rate: int = 16000
    max_audio_seconds: float = 4.0
    # Download pretrained backbone weights (False builds them from their configs).
    pretrained: bool = True
    # Tiny randomly initialised backbones that need no download: for tests and smoke runs.
    tiny: bool = False

    @property
    def max_audio_len(self) -> int:
        return int(self.sample_rate * self.max_audio_seconds)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def tiny_config() -> ModelConfig:
    return ModelConfig(
        text_model="tiny",
        audio_model="tiny",
        vision_model="tiny",
        d_model=32,
        n_heads=2,
        n_layers=1,
        max_text_len=32,
        max_audio_seconds=0.1,
        pretrained=False,
        tiny=True,
    )

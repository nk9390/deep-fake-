from pathlib import Path

import torch

from ml.config import ModelConfig
from ml.models.multimodal_transformer import MultimodalDetector


def save_checkpoint(path, model, **extra):
    """Save weights plus everything needed to rebuild the model offline (plain types only)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": model.cfg.to_dict(),
            "backbone_configs": model.backbone_configs(),
            **extra,
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    # weights_only=True: a .pt file is a pickle, and unpickling an untrusted one can run
    # arbitrary code. This only allows tensors and plain containers.
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = ModelConfig.from_dict(ckpt["config"])
    cfg.pretrained = False
    model = MultimodalDetector(cfg, backbone_configs=ckpt["backbone_configs"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt

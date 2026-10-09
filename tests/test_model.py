import torch

from src.data.loading import build_loader, load_manifest
from src.models.multimodal_transformer import MultimodalDetector
from src.utils.adversarial import feature_attack, input_attack
from src.utils.losses import multitask_loss


def _batch(sample_dir, cfg):
    return next(iter(build_loader(load_manifest(sample_dir / "manifest.csv", cfg), cfg, batch_size=3)))


def test_forward_shapes(sample_dir, cfg):
    model = MultimodalDetector(cfg).eval()
    out = model(_batch(sample_dir, cfg))
    assert out["logits_phishing"].shape == (3, 2) and out["logits_deepfake"].shape == (3, 2)
    assert torch.isfinite(out["logits_phishing"]).all()


def test_missing_modalities_do_not_affect_prediction(sample_dir, cfg):
    torch.manual_seed(0)
    model = MultimodalDetector(cfg).eval()
    batch = _batch(sample_dir, cfg)
    with torch.no_grad():
        before = model(batch)["logits_phishing"]
        # Row 1 has no audio or image: changing those placeholder inputs must not change its output.
        batch["audio_values"][1] = torch.randn_like(batch["audio_values"][1])
        batch["pixel_values"][1] = torch.randn_like(batch["pixel_values"][1])
        after = model(batch)["logits_phishing"]
    assert torch.allclose(before[1], after[1], atol=1e-5)
    assert torch.allclose(before[0], after[0], atol=1e-5)


def test_feature_attack_stays_in_budget_and_raises_loss(sample_dir, cfg):
    torch.manual_seed(0)
    model = MultimodalDetector(cfg).eval()
    batch = _batch(sample_dir, cfg)
    with torch.no_grad():
        features = model.encode(batch)
        clean_loss = multitask_loss(model.classify(features, batch["modality_mask"]), batch)
    delta = feature_attack(model, features, batch, epsilon=0.5, steps=5)
    assert all(d.abs().max() <= 0.5 + 1e-6 for d in delta.values())
    with torch.no_grad():
        adv = {k: v + delta[k] for k, v in features.items()}
        adv_loss = multitask_loss(model.classify(adv, batch["modality_mask"]), batch)
    assert adv_loss > clean_loss


def test_input_attack_perturbs_pixels_within_budget(sample_dir, cfg):
    model = MultimodalDetector(cfg).eval()
    batch = _batch(sample_dir, cfg)
    adv = input_attack(model, batch, ["vision"], epsilon=0.1, steps=2)
    diff = (adv["pixel_values"] - batch["pixel_values"]).abs()
    assert diff.max() <= 0.1 + 1e-6 and diff.max() > 0


def test_freeze_backbones_leaves_only_head_trainable(cfg):
    model = MultimodalDetector(cfg)
    model.freeze_backbones()
    assert model.backbone_parameters() == []
    assert len(model.head_parameters()) > 0

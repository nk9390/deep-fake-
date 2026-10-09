import pytest
import torch

from ml.data.dataset import ManifestDataset, MockMultimodalDataset
from ml.data.loading import build_loader, load_manifest


def test_manifest_parsing(sample_dir, cfg):
    ds = load_manifest(sample_dir / "manifest.csv", cfg)
    assert len(ds) == 3
    full, text_only, image_only = ds[0], ds[1], ds[2]
    assert full["id"] == "full" and full["phishing"] == 1
    assert full["audio"].shape == (int(0.1 * cfg.sample_rate),)  # resampled 8k->16k, capped at 0.1 s
    assert text_only["audio"] is None and text_only["image"] is None and text_only["id"] == "row2"
    assert image_only["id"] == "frame.png" and image_only["phishing"] == -1 and image_only["deepfake"] == 1


def test_missing_file_and_bad_label_are_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        ManifestDataset([{"image": "nope.png"}], root=tmp_path)
    with pytest.raises(ValueError, match="phishing"):
        ManifestDataset([{"text": "hi", "phishing": "yes"}], root=tmp_path)
    with pytest.raises(ValueError, match="at least one"):
        ManifestDataset([{"text": "  "}], root=tmp_path)


def test_unknown_column_rejected(tmp_path):
    (tmp_path / "m.csv").write_text("text,label\nhello,1\n")
    with pytest.raises(ValueError, match="unknown columns"):
        ManifestDataset.from_csv(tmp_path / "m.csv")


def test_mock_is_deterministic():
    a, b = MockMultimodalDataset(4, image_size=16, audio_len=100), MockMultimodalDataset(4, image_size=16, audio_len=100)
    assert a[2]["text"] == b[2]["text"] and torch.equal(a[2]["audio"], b[2]["audio"])


def test_collate_masks_missing_modalities(sample_dir, cfg):
    batch = next(iter(build_loader(load_manifest(sample_dir / "manifest.csv", cfg), cfg, batch_size=3)))
    assert batch["modality_mask"].tolist() == [[True, True, True], [True, False, False], [False, False, True]]
    assert batch["audio_values"].shape == (3, cfg.max_audio_len)
    assert batch["audio_lengths"].tolist() == [cfg.max_audio_len, 0, 0]
    assert batch["pixel_values"].shape == (3, 3, 32, 32)
    assert batch["phishing"].tolist() == [1, 0, -1]

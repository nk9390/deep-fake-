import csv
import json

from src import evaluate, inference, robustness, train
from src.checkpoint import load_checkpoint


def test_train_evaluate_robustness_inference(tmp_path, sample_dir):
    run = tmp_path / "run"
    history = train.main(["--tiny", "--mock_size", "12", "--epochs", "2", "--batch_size", "4",
                          "--output_dir", str(run)])
    assert len(history) == 2 and (run / "best.pt").is_file()
    assert json.loads((run / "history.json").read_text())[0]["epoch"] == 1

    model, ckpt = load_checkpoint(run / "best.pt")
    assert model.cfg.tiny and "metrics" in ckpt

    manifest = str(sample_dir / "manifest.csv")
    metrics = evaluate.main(["--checkpoint", str(run / "best.pt"), "--manifest", manifest])
    assert metrics["phishing"]["n"] == 2 and metrics["deepfake"]["n"] == 3

    report = robustness.main(["--checkpoint", str(run / "best.pt"), "--manifest", manifest,
                              "--epsilons", "0", "0.1", "--steps", "2", "--space", "input"])
    assert [r["epsilon"] for r in report] == [0.0, 0.1]
    assert report[0]["deepfake"]["evasion_rate"] in (None, 0.0)  # no attack at eps=0

    out = tmp_path / "pred.csv"
    rows = inference.main(["--checkpoint", str(run / "best.pt"), "--manifest", manifest, "--output", str(out)])
    assert len(rows) == 3
    with open(out, newline="") as f:
        written = list(csv.DictReader(f))
    assert written[0]["id"] == "full" and "credential_request" in written[0]["red_flags"]
    assert 0.0 <= float(written[0]["phishing_prob"]) <= 1.0


def test_single_sample_inference(tmp_path, sample_dir):
    run = tmp_path / "run"
    train.main(["--tiny", "--mock_size", "4", "--epochs", "1", "--val_fraction", "0", "--adv_weight", "0",
                "--output_dir", str(run)])
    rows = inference.main(["--checkpoint", str(run / "best.pt"), "--text", "Your CEO needs gift cards now",
                           "--image", str(sample_dir / "frame.png"), "--output", str(tmp_path / "one.csv")])
    assert len(rows) == 1 and "payment_request" in rows[0]["red_flags"]

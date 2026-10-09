# Deepfake-Phishing Detector

A multimodal detector for **deepfake-based social engineering**: phishing messages that come
with a cloned voice note or a manipulated photo/video frame ("hi, it's your CEO, here's a voice
memo, buy these gift cards"). It scores each sample on two tasks:

| task | 1 means | signal from |
| --- | --- | --- |
| `phishing` | the message is a social-engineering attempt | mostly text |
| `deepfake` | the audio or image is synthetic/manipulated | mostly audio + image |

Built as a security project, so it also covers the attacker side:

- **Threat model** — who attacks this system and how: [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).
- **Adversarial training** — FGSM perturbations of the modality embeddings during training.
- **Robustness report** — PGD/FGSM evasion attacks on raw pixels, raw audio, or embeddings, with
  the *evasion rate*: how often a malicious sample the detector caught can be pushed past it.
- **Explainable red flags** — rule-based phishing indicators (IP-address and punycode URLs,
  shorteners, urgency, credential/payment requests, authority impersonation) next to every score.
- **Secure engineering** — checkpoints load with `weights_only=True` (no pickle code execution),
  inputs are validated, and text/audio/image reads are size-capped.

## How it works

```
text  ──BERT──────┐
audio ──Wav2Vec2──┼─► 1 token each ─► [CLS] + fusion transformer ─► phishing head
image ──ViT───────┘   (+ modality embedding;                       └► deepfake head
                       missing modalities are masked out)
```

Any modality can be missing (an email has no audio; a voice note has no image). Missing ones
are masked out of the fusion attention, so their placeholder inputs can't affect the prediction.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest -q   # runs offline on tiny random models, ~1 min on CPU
```

Hugging Face backbones (~1.2 GB) download on the first real training run.

## Data

Datasets are CSV manifests. Paths are relative to the manifest file; every column is optional
except that each row needs at least one of `text`, `audio`, `image`. Labels are `1`, `0`, or
empty (unknown — that task is skipped for the row, so partially labelled data works).

```csv
id,text,audio,image,phishing,deepfake
call42,"It's me, wire the money today",clips/call42.wav,,1,1
mail07,"Your invoice is attached",,,0,
face13,,,frames/face13.jpg,,1
```

Audio: any format libsndfile reads (wav/flac/ogg/mp3), mixed to mono, resampled to 16 kHz,
first 4 s used. Video: extract a frame (e.g. `ffmpeg -i in.mp4 -vf "select=eq(n\,0)" frame.jpg`).

Public datasets to build a real manifest from:

| modality | datasets |
| --- | --- |
| deepfake images/video | FaceForensics++, Celeb-DF, DFDC (Kaggle) |
| deepfake audio | ASVspoof 2019/2021 (LA), WaveFake, In-the-Wild |
| phishing text | Nazario phishing corpus, Enron (benign), Kaggle "Phishing Email Dataset" |

There's no public dataset with all three aligned, so a realistic setup is one manifest mixing
rows from each source with the other modalities left empty, plus your own paired samples.

## Usage

```bash
# Smoke test on synthetic data (no download)
python -m src.train --tiny --epochs 2

# Real training
python -m src.train --train_manifest data/train.csv --val_manifest data/val.csv \
    --epochs 5 --batch_size 8 --output_dir outputs/run
#   --freeze_backbones   train only fusion + heads (fast, small GPU)
#   --adv_epsilon 0      disable adversarial training

# Metrics: accuracy, precision, recall, F1, AUC and false-negative rate per task
python -m src.evaluate --checkpoint outputs/run/best.pt --manifest data/test.csv

# Robustness: evasion attacks with growing budgets
python -m src.robustness --checkpoint outputs/run/best.pt --manifest data/test.csv \
    --space vision --epsilons 0 0.01 0.03 0.1 --steps 10
#   --space feature | vision | audio | input (vision+audio)

# Score new samples (writes CSV with probabilities + red flags)
python -m src.inference --manifest data/test/manifest.csv
python -m src.inference --text "URGENT: verify your password at http://192.168.4.1" --audio memo.wav
python -m src.visualize_results          # bar chart -> outputs/plots/predictions.png
```

## Live demo (Cloudflare Pages)

`web/index.html` is a static page: the phishing red-flag analyzer running in the browser (same
rules as the Python module), plus the architecture and threat model. No build step. Two ways to
put it on Cloudflare:

1. **Dashboard, no tokens:** Cloudflare dashboard → Workers & Pages → Create → Pages → Connect
   to Git → pick this repo → framework preset *None*, build command empty, output directory
   `web`. Every push to `main` redeploys.
2. **GitHub Actions:** add repository secrets `CLOUDFLARE_API_TOKEN` (permission *Cloudflare
   Pages: Edit*) and `CLOUDFLARE_ACCOUNT_ID`. `.github/workflows/deploy-pages.yml` then deploys
   to `deep-fake-detector.pages.dev` on every push to `main` that changes `web/`.

The PyTorch model doesn't run on Cloudflare Pages (static hosting, no Python). To serve the
model, run `src.inference` on a machine with Python, or wrap it in a small API on a GPU host.

## Project layout

```
src/
  config.py              model config (saved inside every checkpoint)
  models/                backbones + fusion model
  data/                  CSV manifest + mock datasets, batching
  utils/adversarial.py   FGSM/PGD on embeddings or raw inputs
  utils/phishing_indicators.py   rule-based red flags
  utils/metrics.py       detection metrics
  engine.py              train / predict loops
  train.py evaluate.py robustness.py inference.py visualize_results.py   CLIs
web/index.html           static demo page (Cloudflare Pages)
tests/                   offline tests (tiny models)
docs/THREAT_MODEL.md     attackers, threats, mitigations
docs/REPORT.md           full project report
```

## Limitations

- Trained only on mock data, the model detects nothing real. Its numbers mean something only
  after training on real labelled data.
- One frame per video and one fixed clip length per audio; no temporal modelling.
- Text attacks (paraphrasing, typo-squatting words) aren't in the robustness report yet;
  see the threat model for next steps.

## Report

The full write-up is in **[docs/REPORT.md](docs/REPORT.md)**: problem and real-world cases,
threat model, architecture, adversarial training, robustness methodology (evasion rate), secure
engineering, limitations, future work and references. Its results tables are ready to fill in
once the model is trained on real data.

## Credits

Project concept, original prototype and test data by Karthikey Nori ([@nk9390](https://github.com/nk9390)).
The current implementation (multimodal model, adversarial robustness tooling, threat model,
tests and web demo) was developed with the help of Claude Code, an AI coding assistant.

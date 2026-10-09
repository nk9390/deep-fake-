# Project Report: Adversarially Robust Detection of Deepfake-Based Phishing

**Repository:** https://github.com/nk9390/deep-fake-
**Live demo:** the `web/` page (red-flag analyzer, architecture, threat model), deployable to Cloudflare Pages

---

## Abstract

Phishing used to be judged by its text. Generative AI lets an attacker attach a cloned voice
note or a face-swapped video of a person the victim trusts, which defeats the "does this sound
like my boss?" check people rely on. This project builds a multimodal detector that scores a
message's text, audio and image together for two things: whether it is a social-engineering
attempt (**phishing**) and whether its media is synthetic (**deepfake**). Because a detector is
itself an attack target, the project treats it as a security system: it defines a threat model,
trains with adversarial examples, measures how easily an attacker can evade it (the **evasion
rate**), explains its phishing verdicts with rule-based red flags, and hardens the pipeline
against malicious inputs and model files. The full pipeline (data loading, training,
evaluation, robustness testing, inference, a web demo) is implemented and covered by automated
tests. Detection quality on real attacks still depends on training with real labelled datasets,
which is the main next step.

## 1. Introduction

### 1.1 Problem

Deepfake-enabled fraud is no longer hypothetical:

- **2019:** criminals used an AI-cloned voice of a parent company's chief executive to get the
  head of a UK energy firm to wire €220,000.
- **2024:** an employee of the engineering firm Arup in Hong Kong transferred about US$25 million
  after a video call in which the "CFO" and other colleagues were all deepfakes.

Both attacks pair a classic business-email-compromise request (urgent, authoritative, a
payment) with synthetic media that makes the request believable. Text-only phishing filters
can't see the media, and deepfake detectors don't read the request. This project combines the two.

### 1.2 Objectives

1. Detect phishing and deepfake content from any combination of text, audio and image.
2. Treat the detector as an attack surface: model the attacker, and measure robustness to
   evasion instead of reporting clean accuracy only.
3. Give an analyst an explanation, not only a score.
4. Engineer the pipeline securely: untrusted files must not be able to execute code, exhaust
   resources, or silently corrupt results.

## 2. Threat model (summary)

The full model is in [THREAT_MODEL.md](THREAT_MODEL.md).

| adversary | goal | capability |
| --- | --- | --- |
| Phisher with voice clone / face swap | get a malicious message past the detector | controls every input |
| Adaptive attacker with model access | evasion with minimal perturbation | white-box gradients |
| Data poisoner | backdoor or bias the model | contributes training samples |
| Malicious artifact supplier | code execution on the host | supplies a checkpoint or media file |

Key threats: evasion through small input perturbations (T1), text evasion (T2), dropping the
modality the model handles worst (T3), malicious checkpoints (T4), resource exhaustion (T5),
silent failures (T6), data poisoning (T7), oracle abuse (T8), and new deepfake generators the
model has never seen (T9).

The design choice that follows from this: a **false negative is an attack that got through**,
so the project reports false-negative rate and evasion rate alongside accuracy.

## 3. System design

### 3.1 Architecture

```
text  ──BERT──────┐
audio ──Wav2Vec2──┼─► one token each ─► [CLS] + fusion transformer ─► phishing head (0/1)
image ──ViT───────┘   (+ modality embedding)                        └► deepfake head (0/1)
```

- **Encoders.** BERT (`bert-base-uncased`) for text, Wav2Vec 2.0 (`facebook/wav2vec2-base`) for
  16 kHz audio, and ViT (`google/vit-base-patch16-224`) for images or video frames. Each is
  reduced to one vector: the [CLS] state for text and images, and for audio a mean over only the
  frames that came from real samples, not padding.
- **Fusion.** The three vectors are projected to a shared 512-dimensional space, tagged with a
  learned modality embedding, and passed with a learned [CLS] token through a 4-layer
  pre-norm transformer encoder. Two linear heads read the [CLS] output.
- **Missing modalities.** Real messages rarely carry all three (an email has no audio). Absent
  modalities are excluded through the attention padding mask, so their placeholder inputs
  cannot influence the output. A unit test perturbs the placeholders and checks that the
  prediction doesn't change. This also matters for threat T3: the model can't be made to rely on
  an all-zeros "absent" pattern.
- **Partial labels.** Each task's label may be unknown (`-1`) and is then left out of the loss,
  so datasets that only label deepfakes (FaceForensics++) or only label phishing (email corpora)
  can be mixed in one training set.

### 3.2 Training

- Loss: the sum of the two tasks' cross-entropy over labelled samples.
- Optimiser: AdamW with separate learning rates for the pretrained backbones (2e-5) and the new
  fusion layers and heads (2e-4), gradient clipping at 1.0, and an optional fully frozen-backbone
  mode for small GPUs.
- Model selection: the checkpoint with the lowest validation loss is saved together with its
  full configuration and backbone configs, so it can be rebuilt offline.

### 3.3 Adversarial training

During training each batch is also attacked with the Fast Gradient Sign Method (FGSM;
Goodfellow et al., 2015) in the space of the modality embeddings:

    δ = ε · sign(∇_f L(f, y)),   L_total = (1 − w) · L(f, y) + w · L(f + δ, y)

with ε = 0.05 and w = 0.3 by default. The perturbation is computed on a detached copy, but added
to the live features, so the adversarial term also trains the encoders and not only the fusion
layers.

### 3.4 Explainable red flags

`src/utils/phishing_indicators.py` applies transparent rules to the text and reports which ones
fired:

| indicator | why it matters |
| --- | --- |
| `ip_address_url` | link to a raw IP hides who runs the server |
| `punycode_domain` | `xn--` domains enable look-alikes (pаypal with a Cyrillic а) |
| `credentials_in_url` | `https://bank.com@evil.example/` actually goes to evil.example |
| `url_shortener`, `suspicious_tld`, `unencrypted_link` | hidden or low-trust destinations |
| `urgency`, `credential_request`, `payment_request`, `authority_impersonation` | the social-engineering pattern of the request |

These rules don't feed the model. They give the analyst a reason to trust or doubt a score, and
they're a baseline the learned model should beat. The same rules run in the browser on the demo
page; a check during development confirmed the JavaScript and Python versions give identical
results on the sample messages.

## 4. Robustness evaluation methodology

`src/robustness.py` attacks a trained checkpoint with Projected Gradient Descent (PGD; Madry et
al., 2018), which is FGSM when `--steps 1`, in four places:

| `--space` | what is perturbed | realism |
| --- | --- | --- |
| `vision` | normalised image pixels | realistic: noise added to a deepfake frame |
| `audio` | raw audio samples | realistic: noise added to a cloned voice |
| `input` | pixels and audio together | realistic, strongest input attack |
| `feature` | the modality embeddings | white-box upper bound, not physically realisable |

For each budget ε it reports, per task:

- **clean accuracy** and **adversarial accuracy**, and
- **evasion rate**: of the malicious samples (label 1) that the detector catches on clean input,
  the fraction the attack flips to "benign". This is the number an attacker cares about.

Plotting evasion rate against ε shows how much perturbation it takes to defeat the detector. A
robust model needs a large ε, which means visible artefacts.

## 5. Secure engineering

| risk | control |
| --- | --- |
| A `.pt` checkpoint is a Python pickle; loading an untrusted one can execute code | `torch.load(..., weights_only=True)` only accepts tensors and plain containers; the checkpoint stores configs as JSON strings so this works |
| Oversized inputs exhaust memory | text capped at 20,000 characters; audio reads at most `max_audio_seconds` from the file instead of decoding all of it; PIL's decompression-bomb check on images |
| Missing or corrupt files scored as real ones | the original prototype replaced unreadable images with random noise and still output a prediction; now missing files, unknown columns and invalid labels stop with an error naming the row |
| CSV injection or broken output | results are written with Python's `csv` module, which quotes fields correctly |
| Regressions | 16 automated tests run offline on tiny randomly initialised models and run in GitHub Actions on every push |

## 6. Implementation

| component | file |
| --- | --- |
| model config, saved into every checkpoint | `src/config.py` |
| encoders and fusion model | `src/models/` |
| CSV manifest and mock datasets, batching | `src/data/` |
| FGSM / PGD attacks | `src/utils/adversarial.py` |
| red flags | `src/utils/phishing_indicators.py` |
| metrics (accuracy, precision, recall, F1, AUC, false-negative rate) | `src/utils/metrics.py` |
| command-line tools | `src/train.py`, `evaluate.py`, `robustness.py`, `inference.py`, `visualize_results.py` |
| web demo | `web/index.html` |
| tests and CI | `tests/`, `.github/workflows/ci.yml` |

**Verification so far.** The test suite covers manifest parsing and validation, audio
resampling, missing-modality masking, attack budgets (perturbations stay within ε and raise the
loss), checkpoint round-trips, and the train → evaluate → robustness → inference pipeline end to
end. All tests pass in CI.

## 7. Results

The pipeline has so far been trained only on the synthetic mock dataset, which exists to test
the code. Numbers from it say nothing about real attacks, so none are reported here. To produce
real results:

1. Build manifests from public datasets: FaceForensics++ or Celeb-DF (face deepfakes),
   ASVspoof 2019 LA (voice deepfakes), a phishing corpus such as Nazario plus Enron for
   benign mail. Keep a held-out test split, ideally with one deepfake generator held out
   entirely (threat T9).
2. Train: `python -m src.train --train_manifest data/train.csv --val_manifest data/val.csv`
   (a free Colab GPU is enough with `--freeze_backbones`).
3. Fill in the tables below with `src.evaluate` and `src.robustness`.

**Table 1. Clean detection (test split)**

| task | accuracy | precision | recall | F1 | AUC | false-negative rate |
| --- | --- | --- | --- | --- | --- | --- |
| phishing | | | | | | |
| deepfake | | | | | | |
| red-flag baseline (phishing, score ≥ 0.5) | | | | | — | |

**Table 2. Evasion rate under PGD-10**

| ε | vision (deepfake) | audio (deepfake) | feature (phishing) |
| --- | --- | --- | --- |
| 0.01 | | | |
| 0.03 | | | |
| 0.10 | | | |

**Table 3. Ablation**: the same Table 2 for a model trained with `--adv_epsilon 0`, to measure
what adversarial training buys.

## 8. Limitations

- No real-data results yet (see section 7).
- Adversarial training happens in embedding space only; input-space PGD training is stronger but
  several times slower.
- Text attacks (synonym swaps, homoglyph substitution, zero-width characters) aren't in the
  robustness report yet.
- One frame per video and a fixed audio window: no temporal modelling, so lip-sync and
  voice-to-face mismatches aren't exploited.
- The red-flag rules are English-only and keyword-based, so a careful attacker can avoid them.
- No public dataset has text, audio and image aligned for the same attack, so multimodal fusion
  is trained on mixed single-modality rows plus whatever paired samples are collected.

## 9. Future work

1. Train on the datasets above and complete Tables 1–3.
2. Text attacks with TextAttack, and Unicode normalisation (NFKC, confusable mapping) before the
   text encoder as a defence.
3. Input-space PGD adversarial training.
4. Calibrate the decision threshold for a target false-negative rate instead of a fixed 0.5.
5. Per-generator evaluation and periodic retraining for new deepfake methods.
6. Poisoning defences: provenance tracking, deduplication, spectral-signature outlier checks.

## 10. Conclusion

The project turns a single-model prototype into a multimodal detection pipeline built the way a
security tool should be: with an explicit attacker in mind, robustness measured as an evasion
rate rather than assumed, explanations an analyst can check, and an implementation that
doesn't trust its inputs. The remaining work is empirical: training on real data and filling in
the evaluation tables.

## References

- Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A framework for
  self-supervised learning of speech representations. *NeurIPS*.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep
  bidirectional transformers for language understanding. *NAACL*.
- Dosovitskiy, A., et al. (2021). An image is worth 16x16 words: Transformers for image
  recognition at scale. *ICLR*.
- Goodfellow, I., Shlens, J., & Szegedy, C. (2015). Explaining and harnessing adversarial
  examples. *ICLR*.
- Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards deep learning
  models resistant to adversarial attacks. *ICLR*.
- Rössler, A., et al. (2019). FaceForensics++: Learning to detect manipulated facial images.
  *ICCV*.
- Wang, X., et al. (2020). ASVspoof 2019: A large-scale public database of synthesized, converted
  and replayed speech. *Computer Speech & Language*.

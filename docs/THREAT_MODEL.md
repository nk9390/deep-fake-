# Threat Model

## System

The detector scores an inbound message (text, optional voice note, optional image/video
frame) for **phishing** and **deepfake** content. Its output is meant to warn a user or route
the message to an analyst, not to block it automatically.

## Assets

- The people receiving messages: their credentials, money and trust.
- The detector itself: its weights, its training data, and the host it runs on.

## Adversaries

| adversary | goal | capability |
| --- | --- | --- |
| Phisher using a voice clone / face swap (CEO fraud, "family emergency" scams) | get a malicious message past the detector | crafts all inputs; can query the detector if it's exposed |
| Adaptive attacker with model access (leaked weights, open-source model) | evasion with minimal perturbation | white-box gradients |
| Data poisoner | plant a backdoor or bias | contributes training samples (scraped or crowdsourced data) |
| Malicious artifact supplier | code execution on the host | ships a model checkpoint or a crafted media file |

## Threats and mitigations

| # | threat | where | mitigation in this repo | status |
| --- | --- | --- | --- | --- |
| T1 | **Evasion**: imperceptible noise added to a deepfake image/audio so it scores authentic | inference | FGSM adversarial training; `src.robustness` measures evasion rate on pixels/audio/embeddings | partial: feature-space training only |
| T2 | **Text evasion**: paraphrase, homoglyphs, zero-width chars, typos | inference | rule-based red flags flag homograph (punycode) URLs independently of the model | open: no text attacks in the report yet |
| T3 | **Modality dropping**: send only the modality the model is weakest on | inference | missing modalities are masked, not zero-filled, so each modality must stand alone; train with mixed rows | partial |
| T4 | **Malicious checkpoint**: a `.pt` is a pickle; loading an untrusted one runs code | load | `torch.load(..., weights_only=True)` | mitigated |
| T5 | **Resource exhaustion**: huge text, very long audio, decompression-bomb images | load | text capped at 20k chars; audio reads only the first `max_audio_seconds`; PIL rejects decompression bombs | mitigated |
| T6 | **Silent failure**: missing or corrupt files scored as if they were real | load | missing files and invalid labels raise at load time (the old code substituted random images) | mitigated |
| T7 | **Data poisoning / backdoors** | training | none yet | open: dedupe and audit sources, track provenance, spectral-signature checks |
| T8 | **Model extraction / oracle abuse**: probing an exposed API to tune attacks | deployment | none: don't expose raw probabilities publicly; rate-limit | open |
| T9 | **Distribution shift**: new deepfake generators the model never saw | deployment | none | open: evaluate per generator, retrain regularly |

## Metrics that matter

For a detector, a **false negative is an attack that got through**, so `src.evaluate` reports
the false-negative rate next to accuracy and F1, and `src.robustness` reports the **evasion
rate**: of the malicious samples caught on clean input, the fraction an attacker can flip to
"benign" within a perturbation budget ε.

## Next steps

1. Text attacks with [TextAttack](https://github.com/QData/TextAttack) (synonym swap, homoglyphs).
2. Input-space adversarial training (PGD on pixels/audio), not just embeddings.
3. Per-generator evaluation (hold out one deepfake method entirely).
4. Calibrate the threshold for a target false-negative rate instead of a fixed 0.5.

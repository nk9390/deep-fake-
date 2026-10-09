## Project

Multimodal (text/audio/image) deepfake-phishing detector, framed as a cyber security project
(threat model in docs/THREAT_MODEL.md). Entry points are `python -m src.<train|evaluate|robustness|inference>`.
Run `pytest -q` before pushing; tests use tiny offline models (`tiny_config()`), never downloads.
Keep `torch.load(..., weights_only=True)` and the input validation in `src/data/dataset.py`.

## gstack (recommended)

This project uses [gstack](https://github.com/garrytan/gstack) for AI-assisted workflows.
Install it for the best experience:

```bash
git clone --depth 1 https://github.com/garrytan/gstack.git ~/.claude/skills/gstack
cd ~/.claude/skills/gstack && ./setup --team
```

Skills like /qa, /ship, /review, /investigate, and /browse become available after install.
Use /browse for all web browsing (Aside first, the bundled gstack browser as fallback). Use ~/.claude/skills/gstack/... for gstack file paths.

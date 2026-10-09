## Project

Cyber security toolkit (student project): phishing red flags, email header forensics, fake news
checks with Google's Fact Check API. Core is stdlib-only Python in `cyber/` (`python -m cyber
message|email|news`); web app in `web/` with a Cloudflare Pages Function in `functions/`.
`ml/` is an optional PyTorch deepfake extension; keep the cyber tools independent of it.
The JS rules in `web/index.html` mirror `cyber/phishing.py` and `cyber/news.py`: change both.
Run `pytest -q` before pushing (ML tests only run when torch is installed).
Never put API keys in `web/` or the repo: the fact-check key is a Cloudflare secret.

## gstack (recommended)

This project uses [gstack](https://github.com/garrytan/gstack) for AI-assisted workflows.
Install it for the best experience:

```bash
git clone --depth 1 https://github.com/garrytan/gstack.git ~/.claude/skills/gstack
cd ~/.claude/skills/gstack && ./setup --team
```

Skills like /qa, /ship, /review, /investigate, and /browse become available after install.
Use /browse for all web browsing (Aside first, the bundled gstack browser as fallback). Use ~/.claude/skills/gstack/... for gstack file paths.

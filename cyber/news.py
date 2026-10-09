"""
Rule-based red flags for misleading news. Like the phishing indicators, these explain a verdict
and don't decide truth: a calm, well-written false story passes them, which is why
cyber/factcheck.py checks claims against published fact-checks.
"""
import re
from urllib.parse import urlparse

from cyber.phishing import URL_RE

SENSATIONAL = [
    "shocking", "you won't believe", "exposed", "they don't want you to know", "miracle",
    "secret cure", "banned", "mainstream media won't", "wake up", "100% proof", "bombshell",
    "what happens next", "doctors hate",
]
SHARE_PRESSURE = ["share before", "before it's deleted", "forward this", "send this to everyone", "share this now"]
ATTRIBUTION = [
    "according to", "said", "says", "reported", "study", "survey", "spokesperson",
    "data from", "published in", "told reporters", "statement",
]
# Satire sites whose articles get reshared as real news. Extend with sources you track.
SATIRE_DOMAINS = {"theonion.com", "babylonbee.com", "thebeaverton.com", "newsthump.com", "clickhole.com"}
WORD_RE = re.compile(r"[A-Za-z]{4,}")


def _contains(lowered, phrases):
    return any(re.search(r"\b" + re.escape(p) + r"\b", lowered) for p in phrases)


def _domain(url):
    if not url.lower().startswith(("http://", "https://")):
        url = "http://" + url
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def analyze_news(text: str) -> dict:
    """Return {"indicators": [names...], "score": 0..1} for a headline or article."""
    text = text or ""
    lowered = text.lower()
    indicators = []

    if _contains(lowered, SENSATIONAL):
        indicators.append("sensational_language")
    if _contains(lowered, SHARE_PRESSURE):
        indicators.append("share_pressure")
    words = WORD_RE.findall(text)
    if len(words) >= 4 and sum(w.isupper() for w in words) / len(words) >= 0.3:
        indicators.append("excessive_caps")
    if "!!" in text or text.count("!") >= 3:
        indicators.append("excessive_exclamation")
    if len(text) >= 200 and not _contains(lowered, ATTRIBUTION):
        indicators.append("no_attribution")
    if any(_domain(u) in SATIRE_DOMAINS for u in URL_RE.findall(text)):
        indicators.append("satire_source")

    indicators = sorted(indicators)
    return {"indicators": indicators, "score": round(min(1.0, len(indicators) / 3), 2)}

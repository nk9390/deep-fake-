"""
Rule-based phishing red flags. These don't feed the model; they explain a prediction to an
analyst ("why is this suspicious?") and give a baseline the learned model should beat.
"""
import re
from urllib.parse import urlparse

URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"']+|\b(?:[a-z0-9-]+\.)+[a-z]{2,}/[^\s<>\"']*", re.IGNORECASE)
IP_HOST_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "ow.ly", "rebrand.ly", "cutt.ly"}
SUSPICIOUS_TLDS = {"zip", "mov", "xyz", "top", "click", "country", "gq", "tk", "ml", "cf"}

KEYWORD_GROUPS = {
    "urgency": [
        "urgent", "immediately", "right away", "asap", "within 24 hours", "final notice",
        "suspended", "expire", "locked", "act now",
    ],
    "credential_request": [
        "password", "verify your account", "confirm your", "login", "log in", "otp",
        "one-time code", "2fa code", "security code", "ssn", "social security",
    ],
    "payment_request": [
        "gift card", "wire transfer", "bank details", "payment", "invoice", "bitcoin",
        "crypto", "customs fee", "refund",
    ],
    "authority_impersonation": [
        "ceo", "it department", "help desk", "helpdesk", "irs", "hmrc", "police", "security team",
    ],
}


def _host(url):
    if not url.lower().startswith(("http://", "https://")):
        url = "http://" + url
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def analyze_text(text: str) -> dict:
    """Return {"indicators": [names...], "score": 0..1, "urls": [...]} for a message."""
    text = text or ""
    lowered = text.lower()
    indicators = []

    urls = URL_RE.findall(text)
    for url in urls:
        host = _host(url)
        if IP_HOST_RE.match(host):
            indicators.append("ip_address_url")
        if host.startswith("xn--") or ".xn--" in host:
            indicators.append("punycode_domain")
        if host in SHORTENERS:
            indicators.append("url_shortener")
        if host.rsplit(".", 1)[-1] in SUSPICIOUS_TLDS:
            indicators.append("suspicious_tld")
        if "@" in url.split("//", 1)[-1].split("/", 1)[0]:
            indicators.append("credentials_in_url")
        if url.lower().startswith("http://"):
            indicators.append("unencrypted_link")

    for group, words in KEYWORD_GROUPS.items():
        if any(re.search(r"\b" + re.escape(w) + r"s?\b", lowered) for w in words):
            indicators.append(group)

    indicators = sorted(set(indicators))
    return {"indicators": indicators, "score": round(min(1.0, len(indicators) / 4), 2), "urls": urls}

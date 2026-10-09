"""
Header and link forensics for a raw email (.eml): the checks an analyst runs before trusting
a message, plus the phishing red flags on its subject and body.
"""
import re
from email import policy
from email.parser import BytesParser
from email.utils import getaddresses, parseaddr
from html.parser import HTMLParser
from urllib.parse import urlparse

from cyber.phishing import analyze_text

MAX_EMAIL_BYTES = 5 * 1024 * 1024  # refuse to parse anything larger
AUTH_FAIL_RE = re.compile(r"\b(spf|dkim|dmarc)=(fail|softfail|permerror|temperror|none)\b")
LINKISH_RE = re.compile(r"^(?:https?://)?(?:www\.)?((?:[a-z0-9-]+\.)+[a-z]{2,})(?:[/:?#].*)?$", re.IGNORECASE)

INDICATORS = {
    "spf_fail": "The sending server isn't authorised to send for the From domain (SPF).",
    "dkim_fail": "The message's cryptographic signature is missing or broken (DKIM).",
    "dmarc_fail": "The From domain's own policy check failed (DMARC): a strong sign of spoofing.",
    "reply_to_mismatch": "Replies go to a different domain than the sender's.",
    "return_path_mismatch": "Bounces go to a different domain (common for newsletters, suspicious otherwise).",
    "display_name_spoofing": "The display name contains an email address that isn't the real sender.",
    "link_text_mismatch": "A link's visible text shows one domain but it actually goes to another.",
}


def _domain(address):
    return address.rsplit("@", 1)[-1].lower().strip(">") if "@" in address else ""


def _base(domain):
    """Last two labels: mail.paypal.com and paypal.com both -> paypal.com."""
    return ".".join(domain.split(".")[-2:])


class _LinkCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self._href, self._text = [], None, []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self._href, self._text = dict(attrs).get("href") or "", []

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            self.links.append((self._href, "".join(self._text).strip()))
            self._href = None


def _shown_domain(text):
    """Domain a link's visible text claims to go to, if the text looks like a URL."""
    m = LINKISH_RE.match(text.strip())
    return m.group(1).lower() if m else ""


def _real_host(href):
    href = href.strip()
    if not href.lower().startswith(("http://", "https://")):
        return ""
    try:
        host = (urlparse(href).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def mismatched_links(html):
    """Links whose visible text names a different site than their real destination."""
    collector = _LinkCollector()
    collector.feed(html)
    bad = []
    for href, text in collector.links:
        shown, real = _shown_domain(text), _real_host(href)
        if shown and real and _base(shown) != _base(real):
            bad.append({"text": text, "href": href})
    return bad


def analyze_email(raw: bytes) -> dict:
    if len(raw) > MAX_EMAIL_BYTES:
        raise ValueError(f"Email is larger than {MAX_EMAIL_BYTES // (1024 * 1024)} MB; refusing to parse it")
    msg = BytesParser(policy=policy.default).parsebytes(raw)
    from_name, from_addr = parseaddr(str(msg.get("From", "")))
    from_domain = _domain(from_addr)
    flags = []

    reply_domains = {_domain(a) for _, a in getaddresses([str(v) for v in msg.get_all("Reply-To", [])])} - {""}
    if from_domain and any(_base(d) != _base(from_domain) for d in reply_domains):
        flags.append("reply_to_mismatch")
    return_domain = _domain(parseaddr(str(msg.get("Return-Path", "")))[1])
    if from_domain and return_domain and _base(return_domain) != _base(from_domain):
        flags.append("return_path_mismatch")
    shown = re.search(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", from_name or "")
    if shown and shown.group(0).lower() != from_addr.lower():
        flags.append("display_name_spoofing")

    auth = " ".join(str(v) for v in msg.get_all("Authentication-Results", [])).lower()
    for mechanism, _ in AUTH_FAIL_RE.findall(auth):
        flags.append(f"{mechanism}_fail")

    text_part = msg.get_body(preferencelist=("plain",))
    html_part = msg.get_body(preferencelist=("html",))
    html = html_part.get_content() if html_part else ""
    body = text_part.get_content() if text_part else re.sub(r"<[^>]+>", " ", html)
    links = mismatched_links(html) if html else []
    if links:
        flags.append("link_text_mismatch")

    subject = str(msg.get("Subject", ""))
    return {
        "from": from_addr,
        "subject": subject,
        "header_flags": sorted(set(flags)),
        "mismatched_links": links,
        "content": analyze_text(f"{subject}\n{body}"),
        "authenticated": bool(auth) and not AUTH_FAIL_RE.search(auth),
    }

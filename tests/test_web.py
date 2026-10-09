"""Keep the web app's security headers and its JavaScript rules in sync with the Python code."""
import base64
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = (ROOT / "web" / "index.html").read_text(encoding="utf-8")


def test_csp_hash_matches_inline_script():
    scripts = re.findall(r"<script>(.*?)</script>", PAGE, re.S)
    assert len(scripts) == 1, "CSP allows exactly one inline script"
    digest = base64.b64encode(hashlib.sha256(scripts[0].encode("utf-8")).digest()).decode()
    headers = (ROOT / "web" / "_headers").read_text()
    assert f"'sha256-{digest}'" in headers, f"Update script-src in web/_headers to 'sha256-{digest}'"


def test_no_api_key_in_web_files():
    for path in (ROOT / "web").iterdir():
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"AIza[0-9A-Za-z_-]{30,}", text), f"Google API key committed in {path}"


def test_js_keyword_lists_match_python():
    from cyber import news, phishing

    for words in phishing.KEYWORD_GROUPS.values():
        for w in words:
            assert f'"{w}"' in PAGE, f"{w!r} missing from the web page's phishing rules"
    for w in news.SENSATIONAL + news.SHARE_PRESSURE + news.ATTRIBUTION + sorted(news.SATIRE_DOMAINS):
        assert f'"{w}"' in PAGE, f"{w!r} missing from the web page's news rules"

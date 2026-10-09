# Threat Model

## System

A toolkit that helps a person or a security analyst decide whether to trust a message:

- **Command-line tools** (`cyber/`) that analyse message text, raw emails (`.eml`) and news claims.
- **A public web app** (`web/`) on Cloudflare Pages that runs the phishing and fake news checks
  in the browser.
- **A server function** (`functions/api/factcheck.js`) that looks claims up in Google's Fact
  Check Tools API with a secret API key.

The output is advice ("these red flags fired"), not an automatic block.

## Assets

| asset | why it matters |
| --- | --- |
| Users' credentials, money and trust | what the phisher is after |
| The fact-check API key | lets anyone spend the project's Google quota; tied to the owner's Google account |
| Visitors' browsers | the web app renders text from an external API |
| Integrity of verdicts | a tool that wrongly says "safe" is worse than no tool |

## Adversaries

| adversary | goal | capability |
| --- | --- | --- |
| Phisher | get a scam message or spoofed email believed | writes the whole message, registers domains, sends email from servers they control |
| Misinformation spreader | make a false claim look credible | writes headlines, reposts satire as news |
| Quota abuser | burn the API key's quota or steal the key | calls the public endpoint, reads page source |
| Malicious data source | run script in visitors' browsers | controls text that appears in API results (a claim, a publisher name, a URL) |

## Threats and mitigations

| # | threat | mitigation | status |
| --- | --- | --- | --- |
| T1 | **Rule evasion**: rephrase so no keyword matches, use a fresh domain with HTTPS and no IP address | several independent signals (links, headers, wording); fact-checks don't depend on wording; the optional ML model learns patterns rules miss | partial |
| T2 | **Email spoofing**: fake From address or display name | SPF / DKIM / DMARC results, Reply-To and Return-Path mismatches, display-name spoofing check | detected |
| T3 | **Lying links**: visible text says paypal.com, real target is elsewhere | `link_text_mismatch` compares the text's domain with the real destination | detected |
| T4 | **Homograph domains**: Cyrillic "а" in pаypal.com | `punycode_domain` (the browser version also converts Unicode hosts to `xn--` first) | detected |
| T5 | **API key theft** | key stored as an encrypted Cloudflare secret, used only in the server function, never sent to the browser or logged; upstream errors aren't forwarded because the request URL contains the key | mitigated |
| T6 | **Cross-site scripting through API data** | every API string is HTML-escaped before display; only `http(s)` links are rendered (a `javascript:` URL is dropped server-side and client-side); links use `rel="noopener noreferrer"`; a strict Content-Security-Policy (`web/_headers`) only allows the page's own script by its SHA-256 hash, so injected script can't run even if escaping failed | mitigated |
| T7 | **Quota abuse / denial of service** on `/api/factcheck` | 300-character query cap; identical queries cached at Cloudflare's edge for an hour | partial: add Cloudflare rate limiting |
| T8 | **Server-side request forgery** through the proxy | the function only calls one fixed Google URL; user input goes only into query parameters | mitigated |
| T9 | **Resource exhaustion** with huge inputs | emails over 5 MB are refused; claims are truncated | mitigated |
| T10 | **False reassurance**: "no red flags" or "no fact-checks found" read as "safe" | the UI and CLI say explicitly that no match doesn't mean true or safe | mitigated (by wording) |
| T11 | **Stale knowledge**: brand-new rumours have no fact-check yet | none possible in the tool; advise checking the original source | accepted |
| T12 | **Privacy**: claims typed into the web app are sent to Google | only the fact-check button sends anything; the red-flag checks run locally in the browser | accepted, documented on the page |

## Optional ML extension

The deepfake model in `ml/` adds its own threats: adversarial evasion (invisible noise on a
deepfake image or voice), malicious model checkpoints (a `.pt` file is a pickle), and training
data poisoning. It loads checkpoints with `torch.load(weights_only=True)`, validates and caps
its inputs, and `ml.robustness` measures the evasion rate. See [ml/README.md](../ml/README.md).

## Next steps

1. Cloudflare rate limiting rule on `/api/*` (T7).
2. Unicode confusable normalisation before keyword matching (T1).
3. Domain-age and reputation lookups for links (T1), e.g. via a WHOIS or threat-intel API.

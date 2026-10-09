# Project Report: A Toolkit for Detecting Phishing, Email Spoofing and Fake News

**Repository:** https://github.com/nk9390/deep-fake-

---

## Abstract

Most successful cyber attacks start with a person being tricked, not a system being broken.
Phishing emails, spoofed senders and viral false stories all exploit trust, and generative AI
makes them cheaper to produce and more convincing. This project builds a toolkit that helps a
person or a security analyst decide whether to trust a message. It has three detectors:
**phishing red flags** in text and links, **email header forensics** (SPF, DKIM, DMARC, header
mismatches, display-name spoofing, links whose text lies about their destination), and a
**fake news check** that combines writing red flags with verdicts published by professional
fact-checkers. The tools run from the command line with no dependencies beyond Python, and as
a web app on Cloudflare Pages whose design treats the app itself as an attack target: the API
key never reaches the browser, external data can't inject script, and a strict
Content-Security-Policy backs that up. The project includes a threat model, 19 automated tests,
continuous integration, and an optional machine-learning extension for deepfake voices and
images.

## 1. Introduction

### 1.1 Problem

- **Phishing** is the most common initial step of breaches. Typical tricks: links to raw IP
  addresses or look-alike domains, urgent requests for passwords or payments, and messages
  that impersonate a boss, a bank or IT support.
- **Email spoofing** lets an attacker put any address in the `From:` line. The protocols that
  stop it (SPF, DKIM, DMARC) record their verdicts in headers that ordinary users never see.
- **Fake news** spreads faster than corrections. Professional fact-checkers publish verdicts,
  but readers rarely look them up.
- **Deepfakes** raise the stakes: in 2024 an employee of the engineering firm Arup in Hong Kong
  transferred about US$25 million after a video call in which the "CFO" was a deepfake, and in
  2019 a cloned CEO voice convinced a UK energy firm to wire €220,000. Both started as a
  classic social-engineering request.

### 1.2 Objectives

1. Detect the common technical and linguistic signs of phishing in messages and links.
2. Expose email spoofing by reading the authentication headers users don't see.
3. Check news claims against published fact-checks, not just writing style.
4. Explain every verdict, so the user learns what to look for.
5. Build the web app securely, treating it as an attack surface.

## 2. Threat model (summary)

The full model is in [THREAT_MODEL.md](THREAT_MODEL.md).

| adversary | goal |
| --- | --- |
| Phisher | get a scam message or spoofed email believed |
| Misinformation spreader | make a false claim look credible |
| Quota abuser | steal the API key or burn its quota |
| Malicious data source | run script in visitors' browsers through API results |

Twelve threats are analysed, from rule evasion and spoofing (what the tools detect) to API key
theft, cross-site scripting, request forgery and denial of service (attacks on the tool itself).
A key design principle follows: a **false negative is an attack that got through**, so the
evaluation reports the false-negative rate, and the tools never say "safe", only which red flags
fired.

## 3. Design

### 3.1 Phishing red flags (`cyber/phishing.py`)

Every link in the text is extracted and its host parsed. Then two families of checks run:

| indicator | technique it catches |
| --- | --- |
| `ip_address_url` | link to a raw IP address instead of a domain, hiding who runs the server |
| `punycode_domain` | internationalised `xn--` domains used for homograph look-alikes (pаypal with a Cyrillic а) |
| `credentials_in_url` | `https://bank.com@evil.example/`: the browser ignores everything before `@` |
| `url_shortener` | hidden destination |
| `suspicious_tld` | top-level domains common in abuse reports (`.zip`, `.top`, `.xyz`, ...) |
| `unencrypted_link` | plain `http://` |
| `urgency` | pressure to act before thinking |
| `credential_request` | asks for a password, one-time code or identity number |
| `payment_request` | gift cards, wire transfers, crypto, bank details |
| `authority_impersonation` | claims to be a CEO, IT, a bank or a government agency |

Keyword groups use word-boundary matching (so "ceo" doesn't match inside other words) and accept
plurals. The score is the number of distinct indicators divided by four, capped at 1.

### 3.2 Email forensics (`cyber/email_headers.py`)

The `.eml` file is parsed with Python's standard `email` package and these checks run:

| check | how |
| --- | --- |
| SPF, DKIM, DMARC | read the receiving server's `Authentication-Results` header; `fail`, `softfail`, `none` and error results are flagged |
| Reply-To mismatch | replies would go to a different domain than the sender's |
| Return-Path mismatch | bounces go elsewhere (common for newsletters, so reported as a weaker signal) |
| Display-name spoofing | the display name contains an email address that isn't the real sender, e.g. `"service@paypal.com" <alert@paypa1-support.com>` |
| Lying links | in the HTML part, a link whose visible text is a URL for one site but whose `href` goes to another |

Domains are compared on their last two labels, so `mail.paypal.com` and `paypal.com` count as
the same site. The subject and body also go through the phishing red flags. Emails over 5 MB are
refused before parsing.

### 3.3 Fake news check (`cyber/news.py`, `cyber/factcheck.py`)

**Writing red flags**: sensational or conspiracy wording, pressure to share before checking,
a high share of ALL-CAPS words, repeated exclamation marks, long text with no attribution
("according to", "said", "study"...), and links to known satire sites.

These only catch badly written misinformation: a calm, false story passes them. So the claim is
also looked up in **Google's Fact Check Tools API**, which aggregates ClaimReview verdicts
published by fact-checking organisations such as PolitiFact, Snopes, Reuters, AFP and Full
Fact. Each publisher's free-text rating ("Pants on Fire", "Mostly True", "Missing context") is
mapped to *false*, *true* or *mixed*, checking for mixed wording first so "Half true" isn't
counted as true. The result shows the count per verdict and links to each fact-check.

Two caveats are shown to the user: matches are fact-checks of *similar* claims, and no match
doesn't mean a claim is true, since new rumours haven't been checked yet.

### 3.4 Web app and its security (`web/`, `functions/`)

```
browser ──(claim)──► /api/factcheck  ── Cloudflare Pages Function ──► Google Fact Check API
   ▲                      │  holds FACTCHECK_API_KEY as an encrypted secret
   └── escaped JSON ◄─────┘  returns only cleaned fields, never the upstream URL or body
```

The phishing and news red flags run entirely in the browser (a JavaScript port of the Python
rules), so nothing is sent anywhere until the user clicks *Check fact-checkers*. Security
controls:

| control | threat |
| --- | --- |
| API key only in the server function, stored as a Cloudflare secret | key theft (T5) |
| Upstream errors replaced by generic messages, because the upstream URL contains the key | key leakage through errors (T5) |
| Server keeps only expected fields, truncates strings, drops non-`http(s)` URLs | malicious API data (T6) |
| Client HTML-escapes every string and re-checks link schemes; `rel="noopener noreferrer"` | cross-site scripting (T6), tab-nabbing |
| `Content-Security-Policy` allows only the page's own script by SHA-256 hash, no other script, no framing, no form posting | defence in depth against XSS and clickjacking |
| `X-Content-Type-Options`, `Referrer-Policy: no-referrer`, `Permissions-Policy`, `Cross-Origin-Opener-Policy` | content sniffing, referrer leaks, unwanted device access |
| Function calls one fixed upstream URL; input only in query parameters | server-side request forgery (T8) |
| 300-character query limit; identical queries cached at the edge for an hour | quota abuse (T7) |

Tests keep the security properties from silently breaking: one fails if the inline script
changes without its CSP hash being updated, one fails if anything that looks like a Google API
key is committed to `web/`, and one checks that the JavaScript keyword lists match the Python
ones.

## 4. Implementation

| component | file |
| --- | --- |
| phishing red flags | `cyber/phishing.py` |
| email forensics | `cyber/email_headers.py` |
| fake news red flags | `cyber/news.py` |
| fact-check client | `cyber/factcheck.py` |
| evaluation (precision, recall, F1, false-negative rate) | `cyber/evaluate.py` |
| command line (`python -m cyber message / email / news / evaluate`) | `cyber/__main__.py` |
| web app and security headers | `web/index.html`, `web/_headers` |
| fact-check proxy | `functions/api/factcheck.js` |
| example emails and messages | `samples/` |
| tests | `tests/` |
| optional deepfake ML extension | `ml/` |

Everything in `cyber/` uses only the Python standard library, so it runs on any machine with
Python 3, including lab machines where installing packages isn't allowed.

## 5. Testing and evaluation

### 5.1 Automated tests

19 tests run on every push in GitHub Actions. They cover each phishing indicator, the spoofed
and legitimate sample emails, the lying-link check (including links to raw IP addresses), the
5 MB email limit, rating classification (including "Half true" → mixed), removal of
`javascript:` links from API data, the command line, and the web security checks above. The fact-check
server function was also tested with a mocked API: correct results, clear errors for a missing
query or key, and no key in the error response when Google rejects the request.

### 5.2 Sample results

`samples/phishing.eml` is a spoofed "PayPal" email. The toolkit flags all seven header problems
(SPF, DKIM and DMARC failures, Reply-To and Return-Path mismatches, a display name pretending to
be `service@paypal.com`, and a link that shows `https://www.paypal.com/signin` but goes to
`http://192.168.4.1/login`) plus four content red flags. `samples/legit.eml`, a real-looking
university notice with passing authentication, gets no flags.

On `samples/messages.csv`, ten hand-written messages (five phishing, five legitimate), the
phishing detector at threshold 0.5 scores:

| accuracy | precision | recall | F1 | false-negative rate |
| --- | --- | --- | --- | --- |
| 0.90 | 1.00 | 0.80 | 0.89 | 0.20 |

The one miss is instructive: *"Hey, it's me. Can you lend me some money? I'll explain later."*
It has no link, no keyword and no urgency word, which is exactly the low-tech scam that rules
can't see. This set was written to demonstrate the tool, not to benchmark it, so these numbers
don't predict real-world performance.

### 5.3 How to evaluate on real data

`python -m cyber evaluate data.csv --detector phishing` reads any CSV with `text,label` columns.
Public sources: the Nazario phishing corpus and Enron (legitimate) emails, or the Kaggle
"Phishing Email Dataset"; for news, the LIAR dataset (PolitiFact statements with labels).

| dataset | detector | accuracy | precision | recall | F1 | FNR |
| --- | --- | --- | --- | --- | --- | --- |
| Nazario + Enron sample | phishing | | | | | |
| LIAR test split | news red flags | | | | | |

## 6. Limitations

- **Rules can be evaded.** A careful attacker avoids keywords, uses a fresh domain with HTTPS,
  and sends from a domain whose SPF/DKIM/DMARC they control. The tools catch common and careless
  attacks, not targeted ones.
- **Header checks trust the receiving server.** `Authentication-Results` is written by the
  user's mail provider; a forged header added by the sender before that point could mislead a
  naive parser. The tool reads all such headers, but a production version should only trust
  the one added by the user's own provider.
- **English only.** Keyword lists are in English.
- **Fact-checks lag behind rumours**, and match similar claims, not identical ones.
- **No rate limiting yet** on the public fact-check endpoint beyond caching.

## 7. Optional ML extension

`ml/` contains a multimodal model (BERT for text, Wav2Vec2 for audio, ViT for images, fused by a
transformer) for detecting deepfake voices and images, with adversarial training and an
**evasion-rate** test that measures how often invisible noise flips a caught deepfake to "real".
It applies the same security mindset: model files load with `torch.load(weights_only=True)`,
since a `.pt` file is a pickle that could run code. It isn't needed for the toolkit and isn't
trained on real data yet; see [ml/README.md](../ml/README.md).

## 8. Future work

1. Cloudflare rate limiting on `/api/*`.
2. Only trust the `Authentication-Results` header added by the user's own mail provider.
3. Unicode confusable normalisation before keyword matching (catches "pаssword" with a Cyrillic а).
4. Domain age and reputation lookups (newly registered domains are a strong phishing signal).
5. Evaluate on the public datasets in section 5.3 and tune the threshold for a target
   false-negative rate.
6. Email header analysis in the web app, by pasting raw headers.

## 9. Conclusion

The toolkit turns the checks a security analyst performs (reading authentication headers,
inspecting where links really go, recognising pressure tactics, looking up fact-checks) into
fast, explainable tools anyone can run. Its web app is built on the assumption that it will be
attacked too: secrets stay on the server, external data is treated as hostile, and browser
security headers limit the damage if anything slips through. The main limitation is the one
every rule-based defence has: a careful attacker can write around the rules. That's why the
tools explain their reasoning and never declare a message safe.

## References

- Google. *Fact Check Tools API*. https://developers.google.com/fact-check/tools/api
- Kitterman, S. (2014). *Sender Policy Framework (SPF)*. RFC 7208.
- Crocker, D., Hansen, T., & Kucherawy, M. (2011). *DomainKeys Identified Mail (DKIM) Signatures*. RFC 6376.
- Kucherawy, M., & Zwicky, E. (2015). *Domain-based Message Authentication, Reporting, and Conformance (DMARC)*. RFC 7489.
- Costello, A. (2003). *Punycode*. RFC 3492.
- OWASP. *Cross Site Scripting Prevention Cheat Sheet*. https://cheatsheetseries.owasp.org/
- W3C. *Content Security Policy Level 3*. https://www.w3.org/TR/CSP3/
- Wang, W. Y. (2017). "Liar, liar pants on fire": A new benchmark dataset for fake news detection. *ACL*.

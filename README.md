# Deepfake Phishing Detector

A cyber security toolkit for spotting **social engineering**: phishing messages, spoofed
emails and fake news. Built as a student security project, with a threat model, a project
report, and a web app you can deploy on Cloudflare.

| tool | what it catches | run it |
| --- | --- | --- |
| **Phishing red flags** | IP-address and look-alike (punycode) links, shorteners, `user@host` URL tricks, urgency, password/payment requests, "I'm your CEO" | `python -m cyber message "..."` |
| **Email forensics** | SPF / DKIM / DMARC failures, Reply-To and Return-Path mismatches, spoofed display names, links whose text lies about where they go | `python -m cyber email file.eml` |
| **Fake news check** | sensational wording, share pressure, ALL CAPS, no sources, satire sites; plus **published fact-checks** from PolitiFact, Snopes, Reuters, AFP and others | `python -m cyber news "..."` |
| **Web app** | the phishing and fake news checks in the browser, with a secure fact-check API proxy | `web/` + `functions/` on Cloudflare Pages |

The cyber tools are plain Python 3: no packages to install. An **optional** ML extension for
detecting deepfake voices and images lives in [`ml/`](ml/); you don't need it for anything above.

## Quick start

```bash
git clone https://github.com/nk9390/deep-fake-.git && cd deep-fake-

python -m cyber email samples/phishing.eml     # a spoofed "PayPal" email
python -m cyber email samples/legit.eml        # a clean one, for comparison
python -m cyber message "URGENT: verify your password at http://192.168.4.1/login"
python -m cyber news "SHOCKING!!! Share before it's deleted"
```

Example output for the spoofed email:

```
From: security-alert@paypa1-support.com
Subject: URGENT: your account has been suspended
Sender authentication: failed or missing
Header red flags: display_name_spoofing, dkim_fail, dmarc_fail, link_text_mismatch, reply_to_mismatch, return_path_mismatch, spf_fail
  link shows 'https://www.paypal.com/signin' but goes to http://192.168.4.1/login
Content red flags: credential_request, ip_address_url, unencrypted_link, urgency
```

To check a real email: in Gmail open the message, **⋮ → Download message**; in Outlook
**File → Save As** (.eml). Then run `python -m cyber email` on the file.

### Fact-checking

The news command and the web app search Google's Fact Check Tools API, which collects verdicts
published by fact-checking organisations.

1. In the [Google Cloud console](https://console.cloud.google.com/), create a project, enable
   **Fact Check Tools API**, and create an **API key** (APIs & Services → Credentials). It's free.
2. Command line: `export FACTCHECK_API_KEY=your-key` then `python -m cyber news "claim"`.
3. Web app: add it in Cloudflare (below). Never put the key in `web/` or commit it.

A match is a fact-check of a *similar* claim, so read it. No match doesn't mean a claim is
true: new rumours aren't checked yet.

## Web app on Cloudflare Pages

`web/index.html` is the page; `functions/api/factcheck.js` is a small server function that
calls the fact-check API, so the key never reaches the browser.

1. Cloudflare dashboard → **Compute → Workers & Pages → Create application → Pages →
   Import an existing Git repository** → pick this repo.
2. Production branch `main`, framework preset **None**, build command empty, build output
   directory **`web`** → **Save and Deploy**. Cloudflare finds `functions/` automatically.
3. Project → **Settings → Variables and Secrets → Add**: name `FACTCHECK_API_KEY`, type
   **Secret**, value your Google key. Then **Deployments → Retry deployment** so it takes effect.

You get a public `https://<project>.pages.dev` link, and every push to `main` redeploys it.
Dragging the `web` folder into Cloudflare also works for the page, but uploads that way can't
run the fact-check function.

## Tests

```bash
pip install pytest
pytest -q          # cyber tests; ML tests are skipped unless PyTorch is installed
```

CI runs the cyber tests and the ML tests on every push.

## Project layout

```
cyber/
  phishing.py         phishing red flags in text and links
  email_headers.py    SPF/DKIM/DMARC, header mismatches, display-name spoofing, lying links
  news.py             fake news red flags
  factcheck.py        Google Fact Check API client
  __main__.py         `python -m cyber` command line
functions/api/factcheck.js   Cloudflare server function (keeps the API key secret)
web/index.html               web app
samples/                     example emails
tests/                       cyber tests (tests/ml: ML extension tests)
docs/REPORT.md               project report
docs/THREAT_MODEL.md         threat model
ml/                          optional deepfake ML extension (see ml/README.md)
```

## Report

The full write-up is in **[docs/REPORT.md](docs/REPORT.md)**: the problem with real cases, the
threat model, how each detector works, the secure design of the web app, testing, limitations
and future work.

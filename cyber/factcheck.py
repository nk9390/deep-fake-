"""
Look a claim up in Google's Fact Check Tools API, which aggregates ClaimReview verdicts from
fact-checkers (PolitiFact, Snopes, Reuters, AFP, Full Fact, ...).

    export FACTCHECK_API_KEY=...   # Google Cloud API key with "Fact Check Tools API" enabled
    python -m ml.factcheck "5G towers spread the virus"

The key is read from the environment, never from the command line, so it doesn't end up in
shell history or process listings.
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

from cyber.news import analyze_news

API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
MAX_QUERY_CHARS = 300

FALSE_RATINGS = re.compile(
    r"\b(false|fake|pants on fire|incorrect|inaccurate|misleading|no evidence|fabricated|"
    r"hoax|satire|baseless|unproven|distorts|wrong|scam|altered|not true|four pinocchios)\b",
    re.IGNORECASE,
)
TRUE_RATINGS = re.compile(r"\b(true|correct|accurate|mostly true|verified)\b", re.IGNORECASE)
MIXED_RATINGS = re.compile(
    r"\b(half[- ]true|mixture|mixed|partly|partially|needs context|missing context|cherry[- ]picks)\b", re.IGNORECASE
)


def classify_rating(rating: str) -> str:
    """Map a fact-checker's free-text rating to false / true / mixed."""
    rating = rating or ""
    if MIXED_RATINGS.search(rating):
        return "mixed"
    if FALSE_RATINGS.search(rating):
        return "false"
    if TRUE_RATINGS.search(rating):
        return "true"
    return "mixed"


def safe_url(url):
    """Only pass through http(s) links, so a javascript: URL in API data can't become a link."""
    return url if isinstance(url, str) and url.lower().startswith(("http://", "https://")) else None


def summarize_claims(data: dict) -> dict:
    """Flatten an API response into reviews plus a verdict count."""
    claims, counts = [], {"false": 0, "true": 0, "mixed": 0}
    for claim in data.get("claims", []):
        reviews = []
        for review in claim.get("claimReview", []):
            verdict = classify_rating(review.get("textualRating", ""))
            counts[verdict] += 1
            publisher = review.get("publisher") or {}
            reviews.append({
                "publisher": publisher.get("name") or publisher.get("site") or "unknown",
                "rating": review.get("textualRating", ""),
                "verdict": verdict,
                "title": review.get("title", ""),
                "url": safe_url(review.get("url")),
                "date": review.get("reviewDate", ""),
            })
        claims.append({
            "text": claim.get("text", ""),
            "claimant": claim.get("claimant", ""),
            "date": claim.get("claimDate", ""),
            "reviews": reviews,
        })
    return {"claims": claims, "counts": counts}


def search_claims(query, api_key, language="en", page_size=10, timeout=10):
    query = query.strip()[:MAX_QUERY_CHARS]
    params = urllib.parse.urlencode(
        {"query": query, "languageCode": language, "pageSize": page_size, "key": api_key}
    )
    with urllib.request.urlopen(f"{API_URL}?{params}", timeout=timeout) as resp:
        return json.load(resp)


def main(argv=None):
    p = argparse.ArgumentParser(description="Check a news claim against published fact-checks")
    p.add_argument("claim", help="Headline or claim to check")
    p.add_argument("--language", default="en")
    p.add_argument("--json", action="store_true", help="Print raw summarised JSON")
    args = p.parse_args(argv)

    red_flags = analyze_news(args.claim)
    api_key = os.environ.get("FACTCHECK_API_KEY")
    if not api_key:
        raise SystemExit("Set FACTCHECK_API_KEY (a Google API key with the Fact Check Tools API enabled).")
    try:
        summary = summarize_claims(search_claims(args.claim, api_key, args.language))
    except urllib.error.HTTPError as e:
        # Don't echo the request URL: it contains the key.
        raise SystemExit(f"Fact-check API returned HTTP {e.code}")
    except urllib.error.URLError as e:
        raise SystemExit(f"Could not reach the fact-check API: {e.reason}")

    if args.json:
        print(json.dumps({"red_flags": red_flags, **summary}, indent=2))
        return summary

    print(f"Red flags: {', '.join(red_flags['indicators']) or 'none'}")
    c = summary["counts"]
    if not summary["claims"]:
        print("No published fact-checks match this claim. That doesn't make it true: check the source.")
    else:
        print(f"Fact-checks of similar claims: {c['false']} false, {c['true']} true, {c['mixed']} mixed/other")
        for claim in summary["claims"]:
            print(f"\n- {claim['text']}" + (f"  (claimed by {claim['claimant']})" if claim["claimant"] else ""))
            for r in claim["reviews"]:
                print(f"    {r['publisher']}: {r['rating']}  {r['url'] or ''}")
    return summary


if __name__ == "__main__":
    sys.exit(main() and 0)

"""
One entry point for the cyber tools:

    python -m cyber message "URGENT: verify your password at http://192.168.4.1"
    python -m cyber email suspicious.eml
    python -m cyber news "5G towers spread the virus"          # red flags + fact-checks
    python -m cyber evaluate labelled.csv --detector phishing  # precision / recall
"""
import argparse
import json
import os
import sys
from pathlib import Path

from cyber.email_headers import INDICATORS, analyze_email
from cyber.evaluate import DETECTORS, evaluate_csv
from cyber.news import analyze_news
from cyber.phishing import analyze_text


def _print_flags(title, flags):
    print(f"{title}: {', '.join(flags) if flags else 'none'}")


def cmd_message(args):
    result = analyze_text(args.text)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_flags("Phishing red flags", result["indicators"])
        print(f"Score: {result['score']:.2f}  Links: {', '.join(result['urls']) or 'none'}")
    return result


def cmd_email(args):
    result = analyze_email(Path(args.path).read_bytes())
    if args.json:
        print(json.dumps(result, indent=2))
        return result
    print(f"From: {result['from']}\nSubject: {result['subject']}")
    print(f"Sender authentication: {'passed' if result['authenticated'] else 'failed or missing'}")
    _print_flags("Header red flags", result["header_flags"])
    for flag in result["header_flags"]:
        print(f"  - {flag}: {INDICATORS[flag]}")
    for link in result["mismatched_links"]:
        print(f"  link shows {link['text']!r} but goes to {link['href']}")
    _print_flags("Content red flags", result["content"]["indicators"])
    return result


def cmd_news(args):
    result = analyze_news(args.claim)
    _print_flags("Fake news red flags", result["indicators"])
    if os.environ.get("FACTCHECK_API_KEY"):
        from cyber import factcheck

        factcheck.main([args.claim])
    else:
        print("Set FACTCHECK_API_KEY to also search published fact-checks.")
    return result


def cmd_evaluate(args):
    result = evaluate_csv(args.path, args.detector, args.threshold)
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m cyber", description="Phishing, email and fake news checks")
    sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("message", help="Red flags in a message's text")
    m.add_argument("text")
    m.add_argument("--json", action="store_true")
    m.set_defaults(func=cmd_message)
    e = sub.add_parser("email", help="Header, authentication and link forensics on a .eml file")
    e.add_argument("path")
    e.add_argument("--json", action="store_true")
    e.set_defaults(func=cmd_email)
    n = sub.add_parser("news", help="Fake news red flags, plus fact-checks if FACTCHECK_API_KEY is set")
    n.add_argument("claim")
    n.set_defaults(func=cmd_news)
    v = sub.add_parser("evaluate", help="Precision/recall of a detector on a labelled CSV (text,label)")
    v.add_argument("path")
    v.add_argument("--detector", choices=sorted(DETECTORS), default="phishing")
    v.add_argument("--threshold", type=float, default=0.5)
    v.set_defaults(func=cmd_evaluate)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
    sys.exit(0)

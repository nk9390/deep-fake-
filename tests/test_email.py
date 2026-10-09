from pathlib import Path

import pytest

from cyber.__main__ import main
from cyber.email_headers import MAX_EMAIL_BYTES, analyze_email, mismatched_links

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def test_spoofed_paypal_email_is_flagged():
    result = analyze_email((SAMPLES / "phishing.eml").read_bytes())
    assert set(result["header_flags"]) == {
        "spf_fail", "dkim_fail", "dmarc_fail", "reply_to_mismatch", "return_path_mismatch",
        "display_name_spoofing", "link_text_mismatch",
    }
    assert not result["authenticated"]
    assert result["mismatched_links"][0]["href"] == "http://192.168.4.1/login"
    assert "ip_address_url" in result["content"]["indicators"]


def test_legit_email_is_clean():
    result = analyze_email((SAMPLES / "legit.eml").read_bytes())
    assert result["header_flags"] == [] and result["authenticated"]
    assert result["content"]["indicators"] == []


def test_link_text_matching_same_site_is_fine():
    assert mismatched_links('<a href="https://mail.google.com/x">google.com</a>') == []
    assert mismatched_links('<a href="https://evil.example/">Click here</a>') == []  # text isn't a URL
    assert mismatched_links('<a href="https://evil.example/">www.mybank.com</a>')[0]["text"] == "www.mybank.com"


def test_oversized_email_rejected():
    with pytest.raises(ValueError, match="larger than"):
        analyze_email(b"x" * (MAX_EMAIL_BYTES + 1))


def test_cli(capsys):
    main(["email", str(SAMPLES / "phishing.eml")])
    out = capsys.readouterr().out
    assert "dmarc_fail" in out and "failed or missing" in out
    result = main(["message", "verify your password at http://192.168.4.1", "--json"])
    assert "ip_address_url" in result["indicators"]


def test_evaluate_sample_messages():
    from cyber.evaluate import evaluate_csv

    result = evaluate_csv(SAMPLES / "messages.csv", "phishing", threshold=0.5)
    assert result["n"] == 10
    assert result["true_positives"] + result["false_negatives"] == 5
    assert 0.0 <= result["f1"] <= 1.0

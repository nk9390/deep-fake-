from cyber.factcheck import classify_rating, safe_url, summarize_claims
from cyber.news import analyze_news


def test_flags_typical_fake_news():
    result = analyze_news("SHOCKING!!! Doctors EXPOSED the SECRET CURE they don't want you to know. Share before it's deleted!")
    assert {"sensational_language", "share_pressure", "excessive_caps", "excessive_exclamation"} <= set(result["indicators"])
    assert result["score"] == 1.0


def test_attributed_report_is_clean():
    text = ("The city council approved the new budget on Tuesday, according to a statement from the mayor's office. "
            "The plan raises school funding by 4 percent, officials said, and takes effect in January next year.")
    assert analyze_news(text)["indicators"] == []


def test_long_text_without_sources_and_satire_link():
    text = "x " * 120 + " read more at https://www.theonion.com/some-story"
    assert {"no_attribution", "satire_source"} <= set(analyze_news(text)["indicators"])


def test_classify_rating():
    assert classify_rating("Pants on Fire") == "false"
    assert classify_rating("Misleading") == "false"
    assert classify_rating("Mostly True") == "true"
    assert classify_rating("Half true") == "mixed"
    assert classify_rating("Partly false") == "mixed"
    assert classify_rating("Not true") == "false"
    assert classify_rating("Needs context") == "mixed"


def test_summarize_claims_counts_and_sanitises_links():
    data = {"claims": [{
        "text": "5G spreads viruses",
        "claimant": "social media posts",
        "claimReview": [
            {"publisher": {"name": "Reuters"}, "textualRating": "False", "url": "https://reuters.com/x"},
            {"publisher": {"site": "evil.example"}, "textualRating": "Correct", "url": "javascript:alert(1)"},
        ],
    }]}
    summary = summarize_claims(data)
    assert summary["counts"] == {"false": 1, "true": 1, "mixed": 0}
    reviews = summary["claims"][0]["reviews"]
    assert reviews[0]["publisher"] == "Reuters" and reviews[0]["url"] == "https://reuters.com/x"
    assert reviews[1]["publisher"] == "evil.example" and reviews[1]["url"] is None
    assert safe_url("HTTPS://ok.example") == "HTTPS://ok.example"


def test_summarize_empty_response():
    assert summarize_claims({}) == {"claims": [], "counts": {"false": 0, "true": 0, "mixed": 0}}

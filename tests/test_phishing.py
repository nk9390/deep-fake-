from cyber.phishing import analyze_text


def test_flags_classic_phishing():
    result = analyze_text("URGENT: verify your account at http://192.168.4.1/login or it will be suspended")
    assert {"ip_address_url", "urgency", "credential_request", "unencrypted_link"} <= set(result["indicators"])
    assert result["score"] == 1.0


def test_flags_homograph_and_shortener():
    assert "punycode_domain" in analyze_text("pay at https://xn--pypal-4ve.com/pay")["indicators"]
    assert "url_shortener" in analyze_text("see bit.ly/abc123")["indicators"]
    assert "credentials_in_url" in analyze_text("https://bank.com@evil.example/x")["indicators"]


def test_benign_message_is_clean():
    result = analyze_text("Hi team, the meeting notes from Tuesday are attached.")
    assert result["indicators"] == [] and result["score"] == 0.0


def test_handles_empty_and_none():
    assert analyze_text("")["indicators"] == []
    assert analyze_text(None)["indicators"] == []

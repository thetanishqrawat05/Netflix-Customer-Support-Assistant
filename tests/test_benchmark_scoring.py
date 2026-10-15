from benchmark import score_response

REFUSAL = "I don’t have that information in my support documentation, but I can connect you with a live agent."


def test_refusal_with_curly_apostrophe_counts_for_unanswerable():
    assert score_response("q", False, REFUSAL)


def test_policy_decline_counts_for_unanswerable():
    assert score_response("q", False, "I’m sorry, but I can’t provide anyone else’s password.")


def test_fabricated_answer_fails_for_unanswerable():
    assert not score_response("q", False, "Netflix stock is at $700 today.")


def test_good_answer_mentioning_live_agent_still_passes():
    text = "Refunds are only offered for billing errors. If that's your case, I can connect you with a live agent."
    assert score_response("q", True, text)


def test_refusal_fails_for_answerable():
    assert not score_response("q", True, REFUSAL)

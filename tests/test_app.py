"""End-to-end UI tests with Streamlit's AppTest. The LLM is replaced by a fake
streaming generator so these run offline and deterministically."""
import json
import os

import pytest
from streamlit.testing.v1 import AppTest

import config
import rag_engine

APP = os.path.join(os.path.dirname(__file__), "..", "app.py")
FAKE_RESPONSE = "**Retrieval Reasoning:** Used the two-step verification chunk.\n\n**Answer:** Go to Account > Security & Privacy."


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch, tmp_path):
    def fake_stream(self, query, context, history=None):
        for i in range(0, len(FAKE_RESPONSE), 7):
            yield FAKE_RESPONSE[i:i + 7]

    monkeypatch.setattr(rag_engine.RagEngine, "stream_answer_tokens", fake_stream)
    monkeypatch.setattr(rag_engine.RagEngine, "condense_question", lambda self, q, h=None: q)
    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(config, "FEEDBACK_LOG", str(tmp_path / "feedback.jsonl"))
    monkeypatch.setenv("GROQ_API_KEY", os.getenv("GROQ_API_KEY") or "test-key")
    rag_engine.get_vectordb()  # make sure the engine is warm before the app runs
    return tmp_path


def logged_in_app():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.text_input[0].input("admin")
    at.text_input[1].input("netflix123")
    at.button[0].click()
    at.run()
    assert not at.exception
    assert at.session_state["authenticated"] is True
    return at


def ask(at, question):
    at.chat_input[0].set_value(question)
    at.run()
    assert not at.exception, at.exception


def test_wrong_password_is_rejected():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.text_input[0].input("admin")
    at.text_input[1].input("wrong")
    at.button[0].click()
    at.run()
    assert not at.exception
    assert at.session_state["authenticated"] is False
    assert any("Invalid username or password" in e.value for e in at.error)


def test_non_ascii_credentials_do_not_crash():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.text_input[0].input("ädmin")
    at.text_input[1].input("pässwörd")
    at.button[0].click()
    at.run()
    assert not at.exception
    assert at.session_state["authenticated"] is False


def test_question_is_answered_with_reasoning_split_and_sources():
    at = logged_in_app()
    ask(at, "How do I enable two-step verification?")
    msgs = at.session_state["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    reply = msgs[1]
    assert reply["reasoning"] == "Used the two-step verification chunk."
    assert reply["answer"] == "Go to Account > Security & Privacy."
    assert reply["chunks"] and reply["chunks"][0]["source"] == "02_account_and_security.txt"
    assert reply["ttft"] is not None and reply["latency"] >= reply["ttft"]
    assert any("Go to Account" in m.value for m in at.markdown)


def test_sample_question_chip_asks_question():
    at = logged_in_app()
    next(b for b in at.button if b.label == "Why was my payment declined?").click()
    at.run()
    assert not at.exception
    assert at.session_state["messages"][0]["content"] == "Why was my payment declined?"
    assert "pending_query" not in at.session_state


def test_feedback_is_logged_once_and_cleared_with_chat(fake_llm):
    at = logged_in_app()
    ask(at, "How do I enable two-step verification?")
    for _ in range(2):  # clicking the same rating twice must not log twice
        next(b for b in at.button if b.key == "up_1").click()
        at.run()
    assert at.session_state["feedback_1"] == "up"
    lines = (fake_llm / "feedback.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["rating"] == "up"

    next(b for b in at.sidebar.button if b.label == "Clear chat").click()
    at.run()
    assert at.session_state["messages"] == []
    assert "feedback_1" not in at.session_state


def test_regenerate_replaces_last_answer():
    at = logged_in_app()
    ask(at, "How do I enable two-step verification?")
    next(b for b in at.button if b.key == "regen_1").click()
    at.run()
    assert not at.exception
    msgs = at.session_state["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "How do I enable two-step verification?"


def test_llm_error_is_shown_and_history_stays_consistent(monkeypatch):
    def broken_stream(self, query, context, history=None):
        raise ConnectionError("network down")
        yield  # pragma: no cover

    monkeypatch.setattr(rag_engine.RagEngine, "stream_answer_tokens", broken_stream)
    at = logged_in_app()
    ask(at, "How do I cancel?")
    msgs = at.session_state["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["error"] is True
    assert "internet connection" in msgs[1]["content"]


def test_logout_clears_the_session():
    at = logged_in_app()
    ask(at, "How do I enable two-step verification?")
    next(b for b in at.sidebar.button if b.label == "Log out").click()
    at.run()
    assert not at.exception
    assert at.session_state["authenticated"] is False
    assert "messages" not in at.session_state


def test_transcript_download_available_after_chat():
    at = logged_in_app()
    ask(at, "How do I enable two-step verification?")
    at.run()
    assert len(at.sidebar.download_button) == 1

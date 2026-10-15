import pytest

from response_format import final_sections, split_reasoning


@pytest.mark.parametrize("text", [
    "**Retrieval Reasoning:** Used chunk A.\n\n**Answer:** Go to Account.",
    "Retrieval Reasoning: Used chunk A.\nAnswer: Go to Account.",
    "1. Retrieval Reasoning: Used chunk A.\n2. Answer: Go to Account.",
    "### Retrieval Reasoning\nUsed chunk A.\n### Answer\nGo to Account.",
    "**Retrieval Reasoning**\nUsed chunk A.\n**Answer**\nGo to Account.",
    "**Retrieval Reasoning:**  \nUsed chunk A.\n\n**Answer:**  \nGo to Account.",
    "**Retrieval reasoning**: Used chunk A.\n**answer**: Go to Account.",
])
def test_splits_common_formats(text):
    reasoning, answer, found = split_reasoning(text)
    assert found
    assert reasoning == "Used chunk A."
    assert answer == "Go to Account."


def test_prose_starting_with_answer_word_is_not_a_header():
    text = "**Retrieval Reasoning:** Answering this needs the billing chunk.\nAnswers vary by plan."
    reasoning, answer, found = split_reasoning(text)
    assert not found
    assert answer == ""
    assert reasoning.startswith("Answering this")


def test_partial_stream_before_answer_header():
    reasoning, answer, found = split_reasoning("**Retrieval Reasoning:** Used the billing ch")
    assert (reasoning, answer, found) == ("Used the billing ch", "", False)


def test_final_sections_without_headers_returns_whole_text_as_answer():
    assert final_sections("Just an answer.") == ("", "Just an answer.")


def test_empty_input():
    assert split_reasoning("") == ("", "", False)
    assert final_sections(None) == ("", "")

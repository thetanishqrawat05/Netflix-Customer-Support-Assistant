"""Embedding / retrieval / prompt-history tests. No LLM calls."""
import numpy as np
import pytest

import rag_engine
from config import MAX_HISTORY_TURNS


@pytest.fixture(scope="module")
def engine():
    return rag_engine.RagEngine(temperature=0.0, top_p=0.9, top_k=4)


def test_embeddings_are_normalized_384_dim():
    emb = rag_engine.get_embeddings()
    vecs = np.array(emb.embed_documents(["How do I cancel?", "Buffering issues", "x" * 5000]))
    assert vecs.shape == (3, 384)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    assert emb.embed_documents([]) == []
    assert np.allclose(emb.embed_query("How do I cancel?"), vecs[0], atol=1e-5)


def test_shared_resources_are_built_once():
    assert rag_engine.get_embeddings() is rag_engine.get_embeddings()
    assert rag_engine.get_vectordb() is rag_engine.get_vectordb()
    assert rag_engine.kb_chunk_count() > 0


@pytest.mark.parametrize("question,expected_source", [
    ("How do I enable two-step verification?", "02_account_and_security.txt"),
    ("Why was my payment declined?", "01_billing_and_plans.txt"),
    ("My video keeps buffering", "03_streaming_and_playback.txt"),
    ("What is error code NW-2-5?", "03_streaming_and_playback.txt"),
])
def test_retrieval_finds_the_right_document(engine, question, expected_source):
    prep = engine.prepare(question)
    assert prep["search_query"] == question  # no history -> no rewrite, no LLM call
    assert prep["chunks"][0]["source"] == expected_source
    assert all(0.0 <= c["score"] <= 1.0 for c in prep["chunks"])
    assert prep["top_score"] == max(c["score"] for c in prep["chunks"])
    assert expected_source in prep["context"]


def test_top_k_is_respected_without_rebuilding(engine):
    assert len(engine.retrieve("cancel membership", k=2)) == 2
    assert len(engine.retrieve("cancel membership")) == 4


def test_history_is_trimmed_and_uses_answers_only():
    history = []
    for i in range(MAX_HISTORY_TURNS + 2):
        history.append({"role": "user", "content": f"q{i}"})
        history.append({"role": "assistant", "content": f"**Retrieval Reasoning:** r{i}\n**Answer:** a{i}"})
    msgs = rag_engine.to_history_messages(history)
    assert len(msgs) == MAX_HISTORY_TURNS * 2
    assert msgs[-1].content == f"a{MAX_HISTORY_TURNS + 1}"
    assert all("Retrieval Reasoning" not in m.content for m in msgs)
    assert rag_engine.to_history_messages(None) == []


def test_condense_without_history_skips_llm(engine):
    assert engine.condense_question("How do I cancel?", []) == "How do I cancel?"

import ingest
from config import CHUNK_SIZE


def test_documents_load_in_sorted_order_with_sources():
    docs = ingest.load_documents()
    sources = [d.metadata["source"] for d in docs]
    assert len(docs) >= 4
    assert sources == sorted(sources)
    assert all(d.page_content.strip() for d in docs)


def test_chunks_are_bounded_and_sequentially_numbered():
    chunks = ingest.chunk_documents(ingest.load_documents())
    assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))
    assert all(len(c.page_content) <= CHUNK_SIZE for c in chunks)
    assert all("source" in c.metadata for c in chunks)


def test_fingerprint_changes_when_kb_changes(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    monkeypatch.setattr(ingest, "DATA_DIR", str(tmp_path))
    before = ingest.kb_fingerprint()
    assert before == ingest.kb_fingerprint()
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    assert ingest.kb_fingerprint() != before
    (tmp_path / "b.txt").write_text("new doc", encoding="utf-8")
    assert len({before, ingest.kb_fingerprint()}) == 2


def test_empty_data_dir_loads_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "DATA_DIR", str(tmp_path))
    assert ingest.load_documents() == []
    assert ingest.chunk_documents([]) == []

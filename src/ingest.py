"""Loads raw .txt docs, chunks them, embeds them, and persists to a Chroma vector DB.

Run directly (`python src/ingest.py`) to force a full rebuild. The app also
calls `sync_index()` on startup, which rebuilds automatically when the vector
DB is missing/empty (e.g. a fresh clone) or when files in data/ have changed.
"""
import glob
import hashlib
import json
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

from config import DATA_DIR, VECTOR_DB_DIR, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP

FINGERPRINT_FILE = os.path.join(VECTOR_DB_DIR, "kb_fingerprint.json")


def _kb_files():
    return sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))


def load_documents():
    docs = []
    for path in _kb_files():
        with open(path, encoding="utf-8") as f:
            docs.append(Document(page_content=f.read(), metadata={"source": os.path.basename(path)}))
    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks


def kb_fingerprint():
    """Hash of the knowledge base content + chunking settings."""
    h = hashlib.sha256(f"{CHUNK_SIZE}:{CHUNK_OVERLAP}".encode())
    for path in _kb_files():
        h.update(os.path.basename(path).encode())
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def _stored_fingerprint():
    try:
        with open(FINGERPRINT_FILE, encoding="utf-8") as f:
            return json.load(f).get("fingerprint")
    except (OSError, ValueError):
        return None


def open_vector_store(embeddings):
    # Embeddings are L2-normalized, so the collection uses cosine distance;
    # that keeps Chroma's relevance scores (shown in the UI) within [0, 1].
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_DIR,
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def _uses_cosine(vectordb):
    hnsw = vectordb._collection.configuration.get("hnsw") or {}
    return hnsw.get("space") == "cosine"


def index_is_stale(vectordb):
    return (
        vectordb._collection.count() == 0
        or not _uses_cosine(vectordb)
        or _stored_fingerprint() != kb_fingerprint()
    )


def rebuild_index(vectordb, embeddings):
    """Replaces the collection contents in place. Unlike deleting the
    chroma_db folder, this works while the DB is open (Windows file locks)."""
    if not _uses_cosine(vectordb):
        vectordb.delete_collection()
        vectordb = open_vector_store(embeddings)
    else:
        existing = vectordb.get(include=[])["ids"]
        if existing:
            vectordb.delete(ids=existing)

    chunks = chunk_documents(load_documents())
    if chunks:
        vectordb.add_documents(chunks, ids=[f"chunk-{c.metadata['chunk_id']}" for c in chunks])

    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    with open(FINGERPRINT_FILE, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": kb_fingerprint(), "chunks": len(chunks)}, f)
    return vectordb, len(chunks)


def sync_index(vectordb, embeddings, force=False):
    """Rebuilds only when needed. Returns (vectordb, chunks_written_or_None)."""
    if force or index_is_stale(vectordb):
        return rebuild_index(vectordb, embeddings)
    return vectordb, None


def main():
    from local_embeddings import LocalHFEmbeddings
    from config import EMBEDDING_MODEL

    docs = load_documents()
    print(f"Loaded {len(docs)} source documents from {DATA_DIR}.")
    if not docs:
        print("No .txt files found in data/ - the index will be empty.")

    embeddings = LocalHFEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = open_vector_store(embeddings)
    _, n_chunks = rebuild_index(vectordb, embeddings)
    print(f"Indexed {n_chunks} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) into {VECTOR_DB_DIR}")


if __name__ == "__main__":
    main()

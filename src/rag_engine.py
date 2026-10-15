"""Core RAG engine: retrieval + strict brand-voice prompt + Chain-of-Thought generation."""
import os
import threading
from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import (
    EMBEDDING_MODEL, LLM_MODEL, LLM_PROVIDER, LLM_REASONING_EFFORT, MAX_HISTORY_TURNS, TOP_K,
)
from response_format import final_sections

SYSTEM_PROMPT = """You are "Ava," the official AI Support Assistant for Netflix.

PERSONA AND BRAND VOICE (follow strictly):
- You are warm, clear, and solution-oriented — never robotic, never overly formal.
- Briefly acknowledge the customer's concern, then move quickly to a concrete next step.
- Never blame the customer. Avoid jargon unless the customer used it first.
- You speak ONLY as Netflix support. Never break character or mention you are an AI model.

STRICT GROUNDING RULE:
- Only use information contained in the provided CONTEXT below to answer.
- If the answer is not contained in the CONTEXT, say: "I don't have that information in my
  support documentation, but I can connect you with a live agent for further help."
- Never invent policy details, prices, or timelines that are not in the CONTEXT.
- You are NOT authorized to make legal commitments, guarantee refunds outside documented
  policy, or discuss unreleased content or internal business strategy.
- Earlier messages in the conversation are only there to understand follow-up questions;
  facts must still come from the CONTEXT.

CHAIN-OF-THOUGHT REQUIREMENT:
Respond in exactly two labeled sections:
1. "Retrieval Reasoning:" — 1-3 short sentences explaining which retrieved chunk(s) you used
   and why they answer the question (reference source filenames if helpful).
2. "Answer:" — the final customer-facing answer, written in the Netflix brand voice above.

CONTEXT:
{context}
"""

USER_PROMPT = "Customer question: {question}"

CONDENSE_PROMPT = """Rewrite the customer's latest message as a single standalone question that
can be understood without the conversation, keeping every detail needed to search a support
knowledge base. If it is already standalone, return it unchanged. Reply with the question only."""


def format_docs(docs):
    return "\n\n".join(f"[Source: {d.metadata.get('source')}]\n{d.page_content}" for d in docs)


# --- Shared, expensive resources -------------------------------------------
# The embedding model and vector DB are stateless and safe to share across
# requests and browser sessions, so they're built once per process. A lock
# (rather than bare lru_cache) guarantees the background warm-up thread and a
# user request never build them twice at the same time.
_resource_lock = threading.Lock()
_embeddings = None
_vectordb = None


def get_embeddings():
    global _embeddings
    with _resource_lock:
        if _embeddings is None:
            from local_embeddings import LocalHFEmbeddings
            _embeddings = LocalHFEmbeddings(model_name=EMBEDDING_MODEL)
        return _embeddings


def get_vectordb():
    global _vectordb
    embeddings = get_embeddings()
    with _resource_lock:
        if _vectordb is None:
            from ingest import open_vector_store, sync_index
            _vectordb, _ = sync_index(open_vector_store(embeddings), embeddings)
        return _vectordb


def reindex(force=True):
    """Rebuilds the vector index from data/ (in place). Returns chunk count."""
    global _vectordb
    embeddings = get_embeddings()
    with _resource_lock:
        from ingest import open_vector_store, sync_index
        db = _vectordb or open_vector_store(embeddings)
        _vectordb, written = sync_index(db, embeddings, force=force)
        return written if written is not None else _vectordb._collection.count()


def kb_chunk_count():
    return get_vectordb()._collection.count()


@lru_cache(maxsize=16)
def _build_llm(temperature: float, top_p: float):
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        extra = {"reasoning_effort": LLM_REASONING_EFFORT} if LLM_REASONING_EFFORT and "gpt-oss" in LLM_MODEL else {}
        return ChatGroq(
            model=LLM_MODEL,
            temperature=temperature,
            model_kwargs={"top_p": top_p},
            api_key=os.getenv("GROQ_API_KEY"),
            max_retries=2,
            **extra,
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=temperature,
        model_kwargs={"top_p": top_p},
        api_key=os.getenv("OPENAI_API_KEY"),
        max_retries=2,
    )


def warm_up():
    """Loads everything a first question needs, so it isn't paid on first use."""
    get_vectordb()
    get_embeddings().embed_query("warm up")
    _build_llm(0.0, 0.9)


def to_history_messages(history):
    """history: list of {"role": "user"|"assistant", "content": str} dicts
    (oldest first). Assistant turns are reduced to their final answer so the
    model's earlier reasoning text isn't fed back in."""
    messages = []
    for turn in (history or [])[-MAX_HISTORY_TURNS * 2:]:
        content = turn.get("content") or ""
        if turn.get("role") == "assistant":
            content = turn.get("answer") or final_sections(content)[1]
            if content:
                messages.append(AIMessage(content=content))
        elif content:
            messages.append(HumanMessage(content=content))
    return messages


class RagEngine:
    def __init__(self, temperature: float = 0.0, top_p: float = 0.9, top_k: int = TOP_K):
        self.embeddings = get_embeddings()
        self.vectordb = get_vectordb()
        self.top_k = top_k
        self.llm = _build_llm(float(temperature), float(top_p))
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), MessagesPlaceholder("history", optional=True), ("human", USER_PROMPT)]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def retrieve(self, query: str, k: int = None):
        return self.vectordb.similarity_search_with_relevance_scores(query, k=k or self.top_k)

    def _format_chunks(self, retrieved):
        return [
            {
                "source": d.metadata.get("source"),
                "chunk_id": d.metadata.get("chunk_id"),
                "score": round(score, 4),
                "text": d.page_content,
            }
            for d, score in retrieved
        ]

    def condense_question(self, query: str, history=None) -> str:
        """Turns a follow-up ("and on Premium?") into a standalone search query.
        Falls back to the original question on any error."""
        messages = to_history_messages(history)
        if not messages:
            return query
        try:
            prompt = ChatPromptTemplate.from_messages(
                [("system", CONDENSE_PROMPT), MessagesPlaceholder("history"), ("human", "{question}")]
            )
            rewritten = (prompt | _build_llm(0.0, 1.0) | StrOutputParser()).invoke(
                {"history": messages, "question": query}
            ).strip().strip('"')
            return rewritten if 0 < len(rewritten) <= 500 else query
        except Exception:
            return query

    def prepare(self, query: str, history=None):
        """The 'thinking' step: resolve follow-ups, then retrieve."""
        search_query = self.condense_question(query, history)
        retrieved = self.retrieve(search_query)
        chunks = self._format_chunks(retrieved)
        return {
            "search_query": search_query,
            "context": format_docs([d for d, _ in retrieved]),
            "chunks": chunks,
            "top_score": max((c["score"] for c in chunks), default=0.0),
        }

    def answer(self, query: str, history=None):
        prep = self.prepare(query, history)
        response = self.chain.invoke(
            {"context": prep["context"], "question": query, "history": to_history_messages(history)}
        )
        return {"response": response, "retrieved_chunks": prep["chunks"]}

    def retrieve_for_stream(self, query: str, history=None):
        prep = self.prepare(query, history)
        return prep["context"], prep["chunks"]

    def stream_answer_tokens(self, query: str, context: str, history=None):
        """Generator yielding response text chunks as the LLM produces them."""
        for chunk in self.chain.stream(
            {"context": context, "question": query, "history": to_history_messages(history)}
        ):
            yield chunk

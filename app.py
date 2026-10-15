"""Streamlit chat app for the Netflix Support RAG Assistant.

Mock login in front of the RAG engine, generation-parameter controls, live
"thinking" (retrieval reasoning) and answer streaming, source verification,
conversation memory for follow-ups, feedback logging, and session stats.
"""
import datetime
import hmac
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from config import (
    APP_USERNAME, APP_PASSWORD, DATA_DIR, FEEDBACK_LOG, GENERATION_PRESETS, LLM_MODEL, LLM_PROVIDER,
    LOG_DIR, LOW_CONFIDENCE_THRESHOLD, TOP_K,
)
from response_format import final_sections, split_reasoning

st.set_page_config(page_title="Netflix Support Assistant (Demo)", page_icon="🎬", layout="wide")


# ---------------------------------------------------------------------------
# Background warm-up. Loading the embedding model + vector DB + LLM client
# starts the moment the server first runs this script (i.e. while the user is
# still on the login page), once per process, so logging in never waits on it.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def start_warmup():
    state = {"done": threading.Event(), "error": None, "seconds": None}

    def run():
        started = time.time()
        try:
            import rag_engine
            rag_engine.warm_up()
        except Exception as e:  # surfaced in the sidebar; the next question retries
            state["error"] = f"{type(e).__name__}: {e}"
        finally:
            state["seconds"] = round(time.time() - started, 1)
            state["done"].set()

    threading.Thread(target=run, daemon=True, name="rag-warmup").start()
    return state


warmup = start_warmup()


def engine_ready():
    mod = sys.modules.get("rag_engine")
    return bool(mod and getattr(mod, "_vectordb", None) is not None)


# ---------------------------------------------------------------------------
# Mock Login (Security requirement)
# ---------------------------------------------------------------------------
def credentials_valid(username, password):
    # Constant-time comparison; encode so non-ASCII input can't raise.
    user_ok = hmac.compare_digest(username.strip().encode(), APP_USERNAME.encode())
    pass_ok = hmac.compare_digest(password.encode(), APP_PASSWORD.encode())
    return user_ok and pass_ok


def login_screen():
    st.title("🎬 Netflix Support Assistant")
    st.caption("Internal demo — Mock Login required to access the RAG engine.")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
        if submitted:
            if credentials_valid(username, password):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid username or password.")
    st.info(f"Demo credentials — username: `{APP_USERNAME}` / password: `{APP_PASSWORD}`")


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    login_screen()
    st.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SAMPLE_QUESTIONS = [
    "Why was my payment declined?",
    "How do I enable two-step verification?",
    "My video keeps buffering, what should I do?",
    "Can I get a refund for unused days?",
]

if "messages" not in st.session_state:
    st.session_state.messages = []


def assistant_sections(msg):
    if "answer" in msg:
        return msg.get("reasoning", ""), msg["answer"]
    return final_sections(msg.get("content", ""))


def friendly_error(exc):
    name, text = type(exc).__name__, str(exc).lower()
    if "authentication" in name.lower() or "invalid api key" in text or "401" in text:
        return "The AI service rejected the API key. Please check the key in your `.env` file."
    if "ratelimit" in name.lower() or "429" in text:
        return "The AI service is rate-limiting requests right now. Please wait a moment and try again."
    if "notfound" in name.lower() or "model_not_found" in text:
        return f"The model `{LLM_MODEL}` isn't available for this API key. Set a valid `LLM_MODEL` in `.env`."
    if "connect" in name.lower() or "timeout" in name.lower() or "connection" in text:
        return "Couldn't reach the AI service. Please check your internet connection and try again."
    return "Sorry — something went wrong while answering. Please try again."


def log_feedback(msg_index, rating):
    msgs = st.session_state.messages
    question = msgs[msg_index - 1]["content"] if msg_index > 0 else ""
    record = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "rating": rating,
        "question": question,
        "answer": assistant_sections(msgs[msg_index])[1],
        "sources": [c["source"] for c in msgs[msg_index].get("chunks", [])],
        "params": msgs[msg_index].get("params"),
        "model": LLM_MODEL,
    }
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass  # feedback logging must never break the chat


def build_transcript():
    lines = [f"# Netflix Support Chat — {datetime.datetime.now():%Y-%m-%d %H:%M}", ""]
    for m in st.session_state.messages:
        if m["role"] == "user":
            lines += [f"**Customer:** {m['content']}", ""]
        elif m.get("error"):
            lines += [f"**Ava (error):** {m['content']}", ""]
        else:
            _, answer = assistant_sections(m)
            lines += [f"**Ava:** {answer}", ""]
            sources = sorted({c["source"] for c in m.get("chunks", [])})
            if sources:
                lines += [f"_Sources: {', '.join(sources)}_", ""]
    return "\n".join(lines)


@st.cache_data(show_spinner=False)
def read_kb_file(path, mtime):  # mtime is part of the cache key so edits show up
    with open(path, encoding="utf-8") as f:
        return f.read()


def kb_files():
    return [f for f in sorted(os.listdir(DATA_DIR)) if f.endswith(".txt")] if os.path.isdir(DATA_DIR) else []


def clear_feedback_state():
    for k in [k for k in st.session_state if k.startswith("feedback_")]:
        del st.session_state[k]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🎬 Ava — Netflix Support")
    st.success(f"Logged in as **{APP_USERNAME}**")
    col_logout, col_clear = st.columns(2)
    with col_logout:
        if st.button("Log out", use_container_width=True):
            # Drop the whole session so the next login starts clean.
            st.session_state.clear()
            st.rerun()
    with col_clear:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            # Also drop per-message feedback state so a new answer at a reused
            # index doesn't show a stale 👍/👎 highlight.
            clear_feedback_state()
            st.session_state.pop("pending_query", None)
            st.rerun()

    if engine_ready():
        loaded_in = f" (loaded in {warmup['seconds']}s)" if warmup["seconds"] else ""
        st.caption(f"🟢 AI engine ready{loaded_in}")
    elif warmup["done"].is_set() and warmup["error"]:
        st.caption(f"🔴 Warm-up failed — will retry on your first question. ({warmup['error'][:120]})")
    else:
        st.caption("🟡 AI engine warming up in the background…")

    st.divider()
    st.subheader("Generation Settings")
    preset_name = st.selectbox(
        "Preset",
        list(GENERATION_PRESETS.keys()),
        index=0,
        help="factual_strict (temp=0.0) minimizes hallucination for policy answers; "
             "creative (temp=0.8) is looser and more prone to drifting off the brand voice.",
    )
    preset = GENERATION_PRESETS[preset_name]
    temperature = st.slider("Temperature", 0.0, 1.0, float(preset["temperature"]), 0.05)
    top_p = st.slider("Top-P", 0.1, 1.0, float(preset["top_p"]), 0.05)
    top_k = st.slider("Chunks retrieved (top_k)", 1, 8, TOP_K, 1)
    remember = st.toggle(
        "Remember conversation",
        value=True,
        help="Sends the last few exchanges with each question so follow-ups like "
             "“and on Premium?” are understood. Answers still come only from the knowledge base.",
    )

    st.divider()
    st.caption(f"Model: `{LLM_MODEL}` via **{LLM_PROVIDER}**")
    st.caption(
        "Benchmarking note: temperature=0.0 / top_p=0.9 ('factual_strict') was found to "
        "most reliably stay within the retrieved context and brand voice. See eval/benchmark.py "
        "and the PPT for full results."
    )

    answered = [m for m in st.session_state.messages if m["role"] == "assistant" and not m.get("error")]
    if answered:
        st.divider()
        st.subheader("📊 Session stats")
        ratings = [st.session_state.get(f"feedback_{i}") for i, m in enumerate(st.session_state.messages)
                   if m["role"] == "assistant"]
        c1, c2 = st.columns(2)
        c1.metric("Questions", sum(1 for m in st.session_state.messages if m["role"] == "user"))
        c2.metric("Avg response", f"{sum(m.get('latency', 0) for m in answered) / len(answered):.1f}s")
        ttfts = [m["ttft"] for m in answered if m.get("ttft") is not None]
        c1.metric("Avg first token", f"{sum(ttfts) / len(ttfts):.1f}s" if ttfts else "—")
        c2.metric("👍 / 👎", f"{ratings.count('up')} / {ratings.count('down')}")

    st.divider()
    with st.expander("📚 Browse Knowledge Base"):
        files = kb_files()
        chunk_info = ""
        if engine_ready():
            chunk_info = f" · {sys.modules['rag_engine'].kb_chunk_count()} indexed chunks"
        st.caption(f"{len(files)} documents{chunk_info}. New or edited files in `data/` are re-indexed automatically on startup.")
        if st.button("🔄 Re-index knowledge base", use_container_width=True):
            with st.spinner("Re-indexing…"):
                import rag_engine
                n = rag_engine.reindex(force=True)
            st.toast(f"Knowledge base re-indexed: {n} chunks.", icon="✅")
        for fname in files:
            path = os.path.join(DATA_DIR, fname)
            st.markdown(f"**{fname}**")
            st.text_area(fname, read_kb_file(path, os.path.getmtime(path)), height=120,
                         label_visibility="collapsed")

    if st.session_state.messages:
        st.download_button(
            "⬇️ Download transcript",
            build_transcript(),
            file_name="netflix_support_chat.md",
            mime="text/markdown",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("🎬 Netflix Customer Support Assistant")
st.caption(
    "RAG-powered support bot grounded strictly in Netflix Help Center documentation. "
    "Watch Ava reason over the retrieved sources, then answer — every source is shown for verification."
)

_key_env = "GROQ_API_KEY" if LLM_PROVIDER == "groq" else "OPENAI_API_KEY"
api_key_missing = not os.getenv(_key_env)
if api_key_missing:
    st.warning(
        f"{_key_env} is not set. Add it to a `.env` file (see `.env.example`) "
        "before asking questions.",
        icon="⚠️",
    )


def render_chunks(chunks):
    with st.expander(f"🔍 Retrieved Chunks (source verification) · {len(chunks)}"):
        for c in chunks:
            score = max(0.0, min(1.0, float(c["score"])))
            st.markdown(f"**{c['source']}** · chunk #{c['chunk_id']} · relevance score: `{c['score']}`")
            st.progress(score)
            st.code(c["text"], language=None)


def render_sources_line(chunks):
    best = {}
    for c in chunks:
        best[c["source"]] = max(best.get(c["source"], 0.0), c["score"])
    if best:
        parts = [f"{src} ({score:.0%})" for src, score in sorted(best.items(), key=lambda x: -x[1])]
        st.caption("📎 Sources: " + " · ".join(parts))


def render_actions(msg_index, is_last):
    cols = st.columns([1, 1, 2, 8] if is_last else [1, 1, 10])
    key = f"feedback_{msg_index}"
    current = st.session_state.get(key)
    for col, rating, icon in ((cols[0], "up", "👍"), (cols[1], "down", "👎")):
        with col:
            if st.button(icon, key=f"{rating}_{msg_index}", type=("primary" if current == rating else "secondary")):
                if current != rating:
                    st.session_state[key] = rating
                    log_feedback(msg_index, rating)
                st.rerun()
    if is_last:
        with cols[2]:
            if st.button("🔄 Regenerate", key=f"regen_{msg_index}"):
                msgs = st.session_state.messages
                if msg_index > 0 and msgs[msg_index - 1]["role"] == "user":
                    question = msgs[msg_index - 1]["content"]
                    del msgs[msg_index - 1:]
                    st.session_state.pop(key, None)
                    st.session_state["pending_query"] = question
                st.rerun()


def render_assistant(msg, msg_index, is_last):
    if msg.get("error"):
        st.error(msg["content"])
        if msg.get("detail"):
            with st.expander("Error details"):
                st.code(msg["detail"], language=None)
        render_actions(msg_index, is_last)
        return

    reasoning, answer = assistant_sections(msg)
    if reasoning:
        with st.expander("🧠 How Ava found this (retrieval reasoning)"):
            if msg.get("search_query"):
                st.caption(f"🔎 Searched the knowledge base for: _{msg['search_query']}_")
            st.markdown(reasoning)
    st.markdown(answer)

    chunks = msg.get("chunks") or []
    if chunks:
        if msg.get("top_score") is not None and msg["top_score"] < LOW_CONFIDENCE_THRESHOLD:
            st.caption(
                f"⚠️ Low retrieval confidence (best match {msg['top_score']:.0%}) — this topic may not be "
                "covered by the knowledge base."
            )
        render_sources_line(chunks)
        render_chunks(chunks)
    timing = []
    if msg.get("ttft") is not None:
        timing.append(f"first token {msg['ttft']}s")
    if msg.get("latency") is not None:
        timing.append(f"answered in {msg['latency']}s")
    if timing:
        st.caption("⏱️ " + " · ".join(timing))
    render_actions(msg_index, is_last)


# Show sample question chips only before the conversation starts
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(len(SAMPLE_QUESTIONS))
    for col, q in zip(cols, SAMPLE_QUESTIONS):
        if col.button(q, use_container_width=True):
            st.session_state["pending_query"] = q

last_index = len(st.session_state.messages) - 1
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_assistant(msg, i, is_last=(i == last_index))
        else:
            st.markdown(msg["content"])

query = st.chat_input("Ask about billing, your account, streaming issues, etc.")
if not query and st.session_state.get("pending_query"):
    query = st.session_state.pop("pending_query")
query = (query or "").strip()


def answer_query(question):
    history = list(st.session_state.messages) if remember else []
    history = [m for m in history if not m.get("error")]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.status("🧠 Thinking…", expanded=True)
        with status:
            search_ph = st.empty()
            reasoning_ph = st.empty()
        answer_ph = st.empty()
        started = time.time()
        try:
            if api_key_missing:
                raise RuntimeError(f"{_key_env} is not set")
            if not engine_ready():
                status.update(label="⏳ Loading AI models (first run only, a few seconds)…")
            import rag_engine
            engine = rag_engine.RagEngine(temperature=temperature, top_p=top_p, top_k=top_k)

            status.update(label="🔎 Searching the knowledge base…")
            prep = engine.prepare(question, history)
            if prep["search_query"].strip().lower() != question.lower():
                search_ph.caption(f"🔎 Searched the knowledge base for: _{prep['search_query']}_")
            status.update(label=f"🧠 Reasoning over {len(prep['chunks'])} source chunk(s)…")

            buffer, ttft, answer_started, last_draw = "", None, False, 0.0
            for token in engine.stream_answer_tokens(question, prep["context"], history):
                if not token:
                    continue
                if ttft is None:
                    ttft = round(time.time() - started, 2)
                buffer += token
                now = time.time()
                if now - last_draw < 0.04 and answer_started:
                    continue
                last_draw = now
                reasoning, answer, found = split_reasoning(buffer)
                if found:
                    if not answer_started:
                        answer_started = True
                        reasoning_ph.markdown(reasoning)
                        status.update(label="🧠 Reasoned over the sources — answering…", expanded=False)
                    answer_ph.markdown(answer + " ▌")
                else:
                    reasoning_ph.markdown(reasoning + " ▌")

            if not buffer.strip():
                raise RuntimeError("The model returned an empty response.")
            reasoning, answer = final_sections(buffer)
            if reasoning:
                reasoning_ph.markdown(reasoning)
            else:
                reasoning_ph.empty()
            latency = round(time.time() - started, 2)
            status.update(label=f"🧠 Retrieval reasoning · {len(prep['chunks'])} sources", state="complete",
                          expanded=False)
            answer_ph.markdown(answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": buffer,
                "reasoning": reasoning,
                "answer": answer,
                "chunks": prep["chunks"],
                "top_score": prep["top_score"],
                "search_query": prep["search_query"] if prep["search_query"].strip().lower() != question.lower() else None,
                "ttft": ttft,
                "latency": latency,
                "params": {"temperature": temperature, "top_p": top_p, "top_k": top_k},
            })
        except Exception as e:
            status.update(label="Something went wrong", state="error", expanded=False)
            st.session_state.messages.append({
                "role": "assistant",
                "content": friendly_error(e) if not api_key_missing else
                f"`{_key_env}` is not set. Add it to `.env` and restart the app.",
                "error": True,
                "detail": f"{type(e).__name__}: {e}",
            })


if query:
    answer_query(query)
    st.rerun()

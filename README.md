# Netflix Customer Support RAG Assistant

An AI consultant-style deliverable: a Retrieval-Augmented Generation (RAG) chatbot built
for Netflix's **Internal Customer Support** use case, with a strict brand-voice persona
("Ava"), source-chunk transparency, and a benchmarked comparison of generation parameters.

## Use case

**Customer Support Assistant** — answers subscriber questions about billing, account
security, streaming/playback troubleshooting, and policy, grounded only in Netflix Help
Center-style documentation (`data/`). Out-of-scope or unanswerable questions are refused
rather than hallucinated, per the brand's support escalation policy.

## Quick start

```powershell
.\run.bat
```

On a fresh machine this creates the venv, installs dependencies and creates `.env` (add your
free `GROQ_API_KEY` from console.groq.com/keys, then run it again). The vector DB is built
automatically on first start. Log in with `admin` / `netflix123`.

## Architecture

```
 data/*.txt --chunk--> MiniLM embeddings (ONNX, local) --store--> Chroma (cosine)
                       (auto re-indexed when data/ changes)            |
 follow-up? --LLM rewrite to standalone question--> embed ------------>|
                                                                        v
                                                     top-k retrieved chunks
                                                                        |
               strict brand-voice + CoT prompt + last 3 exchanges       v
                                            --------------------->  LLM (Groq or OpenAI,
                                                                     configurable — see below)
                                                                        |
                                                  streamed token by token
                                                                        v
                         "Retrieval Reasoning" (live "thinking" panel) + "Answer"
                              + sources, relevance scores, low-confidence flag
```

Mock-login-gated Streamlit app (`app.py`) sits in front of the whole pipeline.

## Features

- **Mock login** (constant-time credential check); logout wipes the whole session
- **Live thinking + streaming**: the model's retrieval reasoning streams into a collapsible
  "thinking" panel, then the answer streams below it
- **Conversation memory**: follow-ups ("and if I downgrade?") are rewritten into a standalone
  search query and answered with the last 3 exchanges as context (toggle in the sidebar)
- **Source verification**: per-answer source list with relevance %, full retrieved chunks with
  relevance bars, and a ⚠️ **low-confidence** flag when the best match is weak
- **Generation controls**: presets + temperature / top-p / top-k sliders (no reload on change)
- **Feedback**: 👍/👎 per answer, logged to `logs/feedback.jsonl` for later analysis
- **🔄 Regenerate** the last answer; friendly error messages with details on failure
- **Session stats**: questions asked, average response / first-token time, 👍/👎 counts
- **Knowledge base browser** + one-click **re-index**; new/edited files in `data/` are picked up
  automatically on startup
- **Transcript download** (Markdown, with sources)
- Sample question chips, clear chat, Netflix-style dark theme

## Performance

| | Before | After |
|---|---|---|
| Login → chat page | ~45 s (up to ~2 min cold) | **< 0.1 s** |
| AI engine load (background, starts at the login screen) | ~30–140 s, blocking | **~3–6 s**, non-blocking |
| Changing a slider | full engine rebuild | instant |
| First answer token | ~1.7 s | **~0.6–1.9 s** |

How: embeddings run the official ONNX export of MiniLM via `onnxruntime` instead of PyTorch
(identical vectors, cosine = 1.000000, but no ~30 s `torch`/`transformers` import); the model
loads from the local cache without a Hugging Face network check; the engine warms up in a
background thread while the user is on the login screen and is shared across sessions; and
`reasoning_effort="low"` for gpt-oss (the prompt already makes reasoning explicit).

## Model selection: LLM vs. SLM

| Component | Model class | Choice | Why |
|---|---|---|---|
| Embeddings | SLM | `sentence-transformers/all-MiniLM-L6-v2` (22M params, local, free) | Embedding is a narrow, well-bounded task (semantic similarity) where a small model is already near state-of-the-art for retrieval quality, runs on CPU, costs nothing per call, and keeps customer queries from leaving the local environment until generation — good for a support use case with sensitive account data. |
| Generation | LLM | `openai/gpt-oss-120b` (served via Groq's LPU inference API; `gpt-4o-mini` via OpenAI supported as a drop-in alternative — see `LLM_PROVIDER`/`LLM_MODEL` in `.env`) | Generation here requires: (1) strict persona/brand-voice adherence, (2) multi-sentence Chain-of-Thought reasoning about *why* a chunk was retrieved before answering, (3) graceful refusal reasoning when context is insufficient, and (4) robustness across varied, unpredictable phrasing of customer questions. These are instruction-following and reasoning-heavy requirements that smaller open models (e.g. Phi-3-mini, Llama-3.2-3B) handle less reliably in testing — they more frequently ignore the "refuse if not in context" instruction or break persona under creative-leaning parameters. A large LLM's stronger instruction-following reduces hallucination risk, which matters more for a brand-facing support bot than raw inference cost. Groq was chosen as the serving layer for its very low inference latency (LPU hardware) and generous free tier, which suits an interactive support-chat UX. Note: Groq's available model lineup changes over time — check `console.groq.com/docs/models` if `LLM_MODEL` ever starts returning a "model not found" error. |

**Rule of thumb applied:** use the smallest model that reliably meets the task's reasoning/
instruction-following bar. Embedding = narrow task → SLM. Persona-constrained generation
with refusal logic = LLM.

## Data prep

4 unstructured text documents (`data/`) simulating Netflix Help Center pages:
billing & plans, account & security, streaming/playback troubleshooting, and an internal
brand-voice/escalation policy doc — several pages of realistic unstructured text in total.

## Chunking & embedding

- `RecursiveCharacterTextSplitter`, 500 chars/chunk, 80-char overlap (`src/config.py`)
- Embedded with MiniLM (ONNX Runtime, mean-pooled + L2-normalized, see
  `src/local_embeddings.py`), persisted to a local **Chroma** vector DB configured for
  **cosine distance** (`src/ingest.py`) so the relevance scores shown in the UI stay correctly
  bounded in `[0, 1]`
- The index stores a fingerprint of `data/`; if files change (or the DB is missing, e.g. a fresh
  clone) the app rebuilds the index in place on startup

## Prompt design

- **Strict persona**: "Ava," Netflix's support assistant — see `SYSTEM_PROMPT` in
  `src/rag_engine.py`. Enforces brand tone, grounding-only answers, and documented
  authorization boundaries (no refund guarantees, no legal claims, etc.)
- **Chain-of-Thought**: every response is forced into two labeled sections —
  `Retrieval Reasoning:` (which chunks were used and why) then `Answer:` (the
  customer-facing reply) — so the model's retrieval logic is auditable before the
  final answer is trusted.

## Generation parameters & hallucination findings

See `eval/benchmark.py` (produces `eval/results.csv`) testing 3 presets across
answerable and deliberately out-of-scope questions. On the current 7-question eval set
and this model, all three presets passed the automated refusal/grounding check 7/7 — at
this model's current instruction-following strength, the binary pass/fail check doesn't
distinguish them. The real difference between presets shows up on manual read-through of
`eval/results.csv`, not in the pass rate:

| Preset | Temperature | Top-P | Observed behavior (qualitative) |
|---|---|---|---|
| `factual_strict` | 0.0 | 0.9 | Terse, consistent phrasing; answers stick closely to the retrieved chunk wording. **Recommended for production** for this reason — predictable, low-variance phrasing matters more than tone variety for a billing/policy bot. |
| `balanced` | 0.3 | 0.9 | Slightly more natural phrasing; still grounded, with marginally more rewording of source content. |
| `creative` | 0.8 | 0.95 | Most fluent/varied tone, noticeably slower (higher latency in testing), and most likely of the three to rephrase policy details rather than quote them closely — a higher hallucination *risk* even where this small eval set didn't catch an outright failure. |

**Conclusion documented for the client:** `temperature=0.0` is the right default for a
brand-voice support bot handling billing/policy questions. The automated check here is a
coarse refuse-vs-answer signal, not a fine-grained faithfulness score — a larger eval set
and/or an LLM-as-judge faithfulness check would be needed to quantify the qualitative
differences above with real pass-rate numbers.

## Security

The Streamlit app (`app.py`) requires a **Mock Login** (username/password from `.env`)
before the RAG engine becomes accessible — satisfies the access-control requirement for
this exercise. (Not production auth — no password hashing/session persistence; swap in
a real auth provider, e.g. Auth0 or Streamlit's native auth, before any real deployment.)

## Running it

```powershell
.\run.bat                                   # one command: setup (first run only) + launch
```

Manual equivalent:

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY (free at console.groq.com/keys), optionally change mock login creds

streamlit run app.py           # launch the chat app (builds the vector DB automatically if needed)
python src/ingest.py           # optional: force a full re-index of data/
python eval/benchmark.py       # run the generation-parameter benchmark (live LLM calls)
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests
```

38 offline tests: response parsing, benchmark scoring, ingestion/fingerprinting, embeddings,
retrieval accuracy, conversation-history handling, and end-to-end UI flows via Streamlit's
`AppTest` (login, wrong/non-ASCII credentials, streaming answer, sample chips, feedback logging,
regenerate, error handling, logout, transcript) with a fake LLM.

## Repo layout

```
data/                  raw unstructured knowledge base documents
src/config.py          central config (paths, models, chunking, presets, thresholds)
src/ingest.py          load -> chunk -> embed -> Chroma, plus change detection / in-place re-index
src/local_embeddings.py  MiniLM via ONNX Runtime (no PyTorch)
src/rag_engine.py      shared resources, follow-up rewriting, retrieval, brand-voice/CoT generation
src/response_format.py splits "Retrieval Reasoning" / "Answer" (also while streaming)
app.py                 Streamlit UI: mock login, streaming, sources, feedback, stats
eval/benchmark.py      generation-parameter benchmark harness
tests/                 pytest suite
run.bat                one-command setup + launch
chroma_db/, logs/      generated at runtime (git-ignored)
```

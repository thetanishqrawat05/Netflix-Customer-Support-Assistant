# Netflix Customer Support RAG Assistant — Presentation Content
(Source content for the PPTX deliverable — build via the pptx skill/tool)

## Slide 1 — Title
Netflix Customer Support Assistant: A RAG-Based AI Solution
AI Consultant Engagement — [Your Name], [Date]

## Slide 2 — Business Use Case
- Problem: High support ticket volume for repetitive billing/account/streaming questions
- Solution: A RAG-based "Ava" assistant grounded strictly in Netflix Help Center docs
- Value: Faster resolution, consistent brand voice, auditable answers (shows sources), reduced hallucination risk vs. a bare LLM

## Slide 3 — Architecture Diagram
data/*.txt → chunking (RecursiveCharacterTextSplitter, 500/80) → MiniLM embeddings (ONNX, local) →
Chroma vector DB (cosine; auto re-indexed when data/ changes) → [user query; follow-ups rewritten
into a standalone question] → top-k retrieval → strict brand-voice + CoT prompt + last 3 exchanges →
LLM (Groq: openai/gpt-oss-120b, or OpenAI: gpt-4o-mini), streamed → "Retrieval Reasoning" (live
thinking panel) + "Answer" → Streamlit UI (sources, relevance %, low-confidence flag)
Mock Login gates access to the whole pipeline.
(Use artifact-diagramming or draw.io-style boxes for the actual slide graphic.)

## Slide 4 — Model Choice: LLM vs. SLM
- Embeddings (SLM): all-MiniLM-L6-v2 — narrow task, local, free, fast, keeps data local
- Generation (LLM): openai/gpt-oss-120b via Groq (gpt-4o-mini via OpenAI as alternative) —
  needs strong instruction-following for persona lock, Chain-of-Thought retrieval
  justification, and reliable refusal on out-of-scope queries
- Rule of thumb: smallest model that reliably clears the task's reasoning bar

## Slide 5 — Data Preparation
- 4 unstructured docs: Billing & Plans, Account & Security, Streaming & Playback,
  Internal Brand-Voice/Escalation Policy
- Multi-page realistic unstructured text, simulating a real Help Center KB

## Slide 6 — Chunking & Embedding
- Chunk size 500 chars / 80 overlap — balances context completeness vs. retrieval precision
- MiniLM embeddings (ONNX Runtime — same vectors as PyTorch, ~10x faster to load) in Chroma
- Cosine distance → relevance scores in [0, 1]; best match < 30% is flagged "low confidence"

## Slide 7 — Prompt Design
- Strict persona: "Ava" — Netflix support voice, grounding-only, documented authorization limits
- Chain-of-Thought format: Retrieval Reasoning → Answer (auditable retrieval logic)
- Explicit refusal behavior for out-of-scope / unanswerable questions
- Conversation memory only helps interpret follow-ups; facts must still come from the context

## Slide 8 — Generation Parameter Benchmark
Table (21 live runs: 3 presets × 7 questions):
| Preset | Temp | Top-P | Automated pass rate | Qualitative notes |
|---|---|---|---|---|
| factual_strict | 0.0 | 0.9 | 7/7 | Recommended — terse, consistent, sticks closest to source wording |
| balanced | 0.3 | 0.9 | 7/7 | Slightly more natural rewording, still grounded |
| creative | 0.8 | 0.95 | 7/7 | Most varied tone, slowest; most likely to paraphrase policy details |
Conclusion: the binary refuse/answer check doesn't separate presets on this model; temperature=0.0
is still the production default because low-variance phrasing matters most for billing/policy answers.

## Slide 9 — Streamlit App Demo (screenshots)
- Screenshot 1: Mock Login screen
- Screenshot 2: Live "thinking" panel streaming the retrieval reasoning, then the answer
- Screenshot 3: Sources line + expanded "Retrieved Chunks" panel with relevance bars
- Screenshot 4: A follow-up question ("and if I downgrade?") answered using conversation memory
- Screenshot 5: Sidebar — generation controls, session stats, knowledge base browser

## Slide 10 — Security
- Mock Login gate in front of the RAG engine (constant-time check, env-configured credentials)
- Logout clears the whole session (no chat leaks to the next user of the browser)
- Support assistant never requests passwords/full card numbers/2FA codes in-chat (policy enforced in system prompt)
- Note: demo-grade auth; production would use a real identity provider

## Slide 11 — Evaluation & Engineering Results
- 7-question eval set (4 answerable, 3 deliberately out-of-scope): 21/21 correct across presets
- 38 automated tests (retrieval accuracy, parsing, UI flows) all passing
- Login time ~45 s → < 0.1 s; first answer token ~0.6–1.9 s

## Slide 12 — Conclusion & Next Steps
- RAG approach successfully constrains Netflix's support bot to approved, sourced content
- Next steps: expand KB coverage, LLM-as-judge faithfulness eval, real auth, human-handoff
  integration, and mining logs/feedback.jsonl (👎 + low-confidence questions) to find KB gaps

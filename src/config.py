"""Central configuration for the Netflix Support RAG assistant."""
import os
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load .env from the project root explicitly so it's found no matter which
# directory the app / scripts are launched from.
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# --- Paths ---
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
FEEDBACK_LOG = os.path.join(LOG_DIR, "feedback.jsonl")
COLLECTION_NAME = "netflix_support_kb"

# --- Chunking ---
CHUNK_SIZE = 500          # characters per chunk
CHUNK_OVERLAP = 80        # overlap between chunks to preserve context across boundaries

# --- Embeddings ---
# Local SLM-class embedding model: small, free, runs on CPU, no API cost.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Generation model (the "LLM" in the LLM vs SLM comparison) ---
# LLM provider: "groq" (fast, free-tier friendly, OpenAI-compatible) or "openai".
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

# Groq-hosted gpt-oss-120b chosen as a cost-efficient, low-latency LLM with
# strong instruction-following, needed for Chain-of-Thought-style retrieval
# justification and nuanced brand-voice tone control. See README for the
# full LLM vs SLM justification.
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b" if LLM_PROVIDER == "groq" else "gpt-4o-mini")

# gpt-oss models on Groq do hidden reasoning before answering. Our prompt
# already makes the model write its retrieval reasoning out explicitly, so
# "low" keeps quality while cutting time-to-first-token ~1.6s -> ~0.6s.
# Set to "" to use the provider default.
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "low")

# --- Retrieval ---
TOP_K = 4  # number of chunks retrieved per query

# Best-match relevance below this is flagged as "low confidence" in the UI.
# Calibrated on this KB: in-scope questions score >= ~0.38, off-topic <= ~0.14.
LOW_CONFIDENCE_THRESHOLD = 0.30

# --- Conversation memory ---
# Number of previous user/assistant exchanges sent with each question so
# follow-ups like "what about on Premium?" make sense.
MAX_HISTORY_TURNS = 3

# --- Generation parameter presets used for benchmarking (see eval/benchmark.py) ---
GENERATION_PRESETS = {
    "factual_strict": {"temperature": 0.0, "top_p": 0.9},
    "balanced": {"temperature": 0.3, "top_p": 0.9},
    "creative": {"temperature": 0.8, "top_p": 0.95},
}

# --- Mock auth ---
APP_USERNAME = os.getenv("APP_USERNAME", "admin")
APP_PASSWORD = os.getenv("APP_PASSWORD", "netflix123")

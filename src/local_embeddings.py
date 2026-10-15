"""Lightweight local embedding wrapper for sentence-transformers/all-MiniLM-L6-v2.

Runs the model's official ONNX export through `onnxruntime` + `tokenizers`
instead of `torch` + `transformers`. Same weights, same mean pooling and L2
normalization as sentence-transformers, but importing it costs well under a
second instead of the ~30s (warm) to ~2min (cold) that torch + transformers
take on this machine, which is what made the first login so slow. It also
avoids sentence-transformers' scipy/sklearn dependency, whose compiled
extensions are blocked by this machine's Windows Application Control policy.
"""
import os
from typing import List

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from langchain_core.embeddings import Embeddings

MAX_SEQ_LENGTH = 256  # matches sentence-transformers' setting for this model
BATCH_SIZE = 32


def _model_file(model_name: str, filename: str) -> str:
    # Use the local HF cache without a network round trip when possible; only
    # reach out to the Hub the very first time a file isn't cached yet.
    try:
        return hf_hub_download(model_name, filename, local_files_only=True)
    except Exception:
        return hf_hub_download(model_name, filename)


class LocalHFEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = Tokenizer.from_file(_model_file(model_name, "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            _model_file(model_name, "onnx/model.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {i.name for i in self.session.get_inputs()}

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)

        last_hidden_state = self.session.run(None, feeds)[0]
        mask = attention_mask[..., None].astype(np.float32)
        pooled = (last_hidden_state * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled / np.clip(norms, 1e-12, None)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = [self._embed_batch(texts[i:i + BATCH_SIZE]) for i in range(0, len(texts), BATCH_SIZE)]
        return np.vstack(vectors).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(list(texts))

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

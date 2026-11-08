"""Embedding providers: Ollama (default) and an offline hashed n-gram embedder."""

from __future__ import annotations

import hashlib
from typing import Protocol

import httpx
import numpy as np

from .textnorm import tokenize


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized vectors."""
        ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class OllamaEmbedder:
    """Calls Ollama's ``POST /api/embed`` (batch input)."""

    def __init__(self, base_url: str, model: str, timeout: float = 120.0, client: httpx.Client | None = None):
        self.model = model
        self.name = f"ollama:{model}"
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout)

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            try:
                resp = self._client.post("/api/embed", json={"model": self.model, "input": batch})
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise RuntimeError(
                    f"Ollama embedding request failed ({exc}). Is Ollama running and is "
                    f"'{self.model}' pulled? Try: ollama pull {self.model}"
                ) from exc
            data = resp.json().get("embeddings")
            if not data or len(data) != len(batch):
                raise RuntimeError(f"Unexpected embedding response from Ollama: {resp.text[:200]}")
            vectors.extend(data)
        return l2_normalize(np.asarray(vectors, dtype=np.float32))


class HashEmbedder:
    """Deterministic, dependency-free embedder based on hashed character n-grams
    of normalized tokens (stopwords dropped, punctuation ignored). Not semantic,
    but robust to Arabic morphology and good enough for tests and offline demos."""

    def __init__(self, dim: int = 512, ngram: tuple[int, ...] = (3, 4)):
        self.dim = dim
        self.ngram = ngram
        self.name = f"hash:{dim}"

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for word in tokenize(text):
            padded = f" {word} "
            for n in self.ngram:
                for i in range(max(1, len(padded) - n + 1)):
                    gram = padded[i:i + n]
                    h = int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=8).digest(), "little")
                    vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.stack([self._vector(t) for t in texts]))


def make_embedder(settings) -> Embedder:
    if settings.embedder == "hash":
        return HashEmbedder()
    if settings.embedder == "ollama":
        return OllamaEmbedder(settings.ollama_base_url, settings.embed_model, settings.llm_timeout)
    raise ValueError(f"Unknown SANAD_EMBEDDER: {settings.embedder!r} (use 'ollama' or 'hash')")

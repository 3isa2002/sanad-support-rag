"""Hybrid (dense + BM25) index over KB chunks, persisted to disk."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .embeddings import Embedder
from .kb import Chunk
from .textnorm import tokenize

RRF_K = 60
SAME_LANG_BONUS = 0.5 / (RRF_K + 1)  # half of a rank-1 vote


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # dense cosine similarity, used as the confidence signal
    bm25: float
    fused: float


class HybridIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder_name: str):
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        self.chunks = chunks
        self.vectors = vectors.astype(np.float32)
        self.embedder_name = embedder_name
        self.bm25 = BM25([tokenize(c.embed_text()) for c in chunks])

    # ---------- build / persist ----------
    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> "HybridIndex":
        if not chunks:
            raise ValueError("No chunks to index — is the KB directory empty?")
        vectors = embedder.embed([c.embed_text() for c in chunks])
        return cls(chunks, vectors, embedder.name)

    def save(self, index_dir: Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "vectors.npy", self.vectors)
        (index_dir / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False, indent=1), encoding="utf-8"
        )
        (index_dir / "meta.json").write_text(
            json.dumps({"embedder": self.embedder_name, "count": len(self.chunks)}), encoding="utf-8"
        )

    @classmethod
    def load(cls, index_dir: Path, expected_embedder: str | None = None) -> "HybridIndex":
        index_dir = Path(index_dir)
        if not (index_dir / "meta.json").is_file():
            raise FileNotFoundError(f"No index found in {index_dir}. Run `sanad ingest` first.")
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        if expected_embedder and meta["embedder"] != expected_embedder:
            raise ValueError(
                f"Index was built with '{meta['embedder']}' but the current embedder is "
                f"'{expected_embedder}'. Re-run `sanad ingest`."
            )
        chunks = [Chunk(**c) for c in json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))]
        vectors = np.load(index_dir / "vectors.npy")
        return cls(chunks, vectors, meta["embedder"])

    # ---------- search ----------
    def search(self, query: str, embedder: Embedder, top_k: int = 4, lang: str | None = None) -> list[SearchResult]:
        qvec = embedder.embed([query])[0]
        dense = self.vectors @ qvec
        lexical = np.asarray(self.bm25.scores(tokenize(query)), dtype=np.float32)

        fused = np.zeros(len(self.chunks), dtype=np.float32)
        for ranking in (np.argsort(-dense), np.argsort(-lexical)):
            for rank, idx in enumerate(ranking):
                fused[idx] += 1.0 / (RRF_K + rank + 1)
        # Lexical ranks only count where there was an actual term match.
        no_match = lexical <= 0
        if no_match.any():
            lex_rank = np.empty(len(lexical), dtype=np.int64)
            lex_rank[np.argsort(-lexical)] = np.arange(len(lexical))
            fused[no_match] -= 1.0 / (RRF_K + lex_rank[no_match] + 1)
        if lang:
            fused += np.array([SAME_LANG_BONUS if c.lang == lang else 0.0 for c in self.chunks], dtype=np.float32)

        order = np.argsort(-fused)[:top_k]
        return [
            SearchResult(self.chunks[i], float(dense[i]), float(lexical[i]), float(fused[i]))
            for i in order
        ]

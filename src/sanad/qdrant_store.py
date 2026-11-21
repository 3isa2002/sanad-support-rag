"""Qdrant-backed hybrid vector store (dense embeddings + Okapi BM25 sparse vectors + RRF)."""

from __future__ import annotations

import atexit
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from .bm25 import BM25
from .embeddings import Embedder
from .index import HybridIndex, SAME_LANG_BONUS, SearchResult
from .kb import Chunk, build_chunks
from .textnorm import tokenize

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "lexical"


def _sparse_doc_vector(
    freqs: Counter[str],
    dl: int,
    bm25: BM25,
    vocab: dict[str, int],
) -> models.SparseVector:
    """Compute exact Okapi BM25 document term weights as a Qdrant sparse vector."""
    indices: list[int] = []
    values: list[float] = []
    avgdl = bm25.avgdl or 1.0
    for term, tf in freqs.items():
        idx = vocab.get(term)
        if idx is None:
            continue
        idf = bm25.idf.get(term, 0.0)
        denom = tf + bm25.k1 * (1.0 - bm25.b + bm25.b * dl / avgdl)
        weight = idf * tf * (bm25.k1 + 1.0) / denom
        if weight > 0:
            indices.append(idx)
            values.append(float(weight))
    return models.SparseVector(indices=indices, values=values)


def _sparse_query_vector(tokens: list[str], vocab: dict[str, int]) -> models.SparseVector:
    """Map query tokens to sparse vector weights matching ``BM25.scores`` (1.0 per term occurrence)."""
    counts = Counter(t for t in tokens if t in vocab)
    indices = [vocab[t] for t in counts]
    values = [float(c) for c in counts.values()]
    return models.SparseVector(indices=indices, values=values)


def _build_client(
    *,
    qdrant_url: str = "",
    qdrant_path: Path | str | None = None,
    in_memory: bool = False,
) -> QdrantClient:
    if in_memory:
        return QdrantClient(":memory:")
    if qdrant_url:
        return QdrantClient(url=qdrant_url)
    target = Path(qdrant_path) if qdrant_path is not None else Path("data/index/qdrant")
    target.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(target))


class QdrantVectorIndex:
    """Hybrid dense + sparse vector store backed by Qdrant (:memory:, local disk, or server)."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        chunks: list[Chunk],
        embedder_name: str,
        *,
        qdrant_url: str = "",
        qdrant_path: Path | None = None,
    ):
        self.client = client
        self.collection_name = collection_name
        self.chunks = chunks
        self.embedder_name = embedder_name
        self.qdrant_url = qdrant_url
        self.qdrant_path = qdrant_path
        self._chunk_by_idx = {i: c for i, c in enumerate(chunks)}
        tokenized = [tokenize(c.embed_text()) for c in chunks]
        self.bm25 = BM25(tokenized)
        terms = sorted(self.bm25.idf.keys())
        self.vocab: dict[str, int] = {t: i for i, t in enumerate(terms)}
        self._closed = False
        atexit.register(self.close)

    def close(self) -> None:
        """Release local Qdrant file locks cleanly (especially important on Windows)."""
        if self._closed:
            return
        self._closed = True
        try:
            self.client.close()
        except Exception:
            pass

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        embedder: Embedder,
        *,
        collection_name: str = "sanad_kb",
        qdrant_url: str = "",
        qdrant_path: Path | str | None = None,
        in_memory: bool = False,
        client: QdrantClient | None = None,
    ) -> "QdrantVectorIndex":
        """Embed chunks and populate a fresh Qdrant hybrid collection (dense + lexical sparse)."""
        if not chunks:
            raise ValueError("No chunks to index — is the KB directory empty?")
        vectors = embedder.embed([c.embed_text() for c in chunks]).astype(np.float32)
        resolved_path = Path(qdrant_path) if qdrant_path is not None else None
        q_client = client or _build_client(
            qdrant_url=qdrant_url,
            qdrant_path=resolved_path,
            in_memory=in_memory,
        )

        instance = cls(
            q_client,
            collection_name,
            chunks,
            embedder.name,
            qdrant_url=qdrant_url,
            qdrant_path=resolved_path,
        )

        if q_client.collection_exists(collection_name):
            q_client.delete_collection(collection_name)

        dim = int(vectors.shape[1])
        q_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(size=dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams()},
        )

        points = []
        for idx, (chunk, vec, freqs, dl) in enumerate(
            zip(chunks, vectors, instance.bm25.doc_freqs, instance.bm25.doc_lens)
        ):
            payload = {**chunk.to_dict(), "chunk_index": idx, "embedder": embedder.name}
            points.append(
                models.PointStruct(
                    id=idx,
                    vector={
                        DENSE_VECTOR_NAME: vec.tolist(),
                        SPARSE_VECTOR_NAME: _sparse_doc_vector(freqs, dl, instance.bm25, instance.vocab),
                    },
                    payload=payload,
                )
            )
        q_client.upsert(collection_name=collection_name, points=points, wait=True)
        return instance

    def save(self, index_dir: Path) -> None:
        """Persist collection metadata alongside the local Qdrant directory."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "backend": "qdrant",
            "embedder": self.embedder_name,
            "collection": self.collection_name,
            "count": len(self.chunks),
            "qdrant_url": self.qdrant_url,
            "qdrant_path": str(self.qdrant_path) if self.qdrant_path is not None else "",
        }
        (index_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        index_dir: Path,
        *,
        expected_embedder: str | None = None,
        collection_name: str = "sanad_kb",
        qdrant_url: str = "",
        qdrant_path: Path | str | None = None,
    ) -> "QdrantVectorIndex":
        """Open an existing Qdrant collection and validate its embedder matches."""
        index_dir = Path(index_dir)
        meta_file = index_dir / "meta.json"
        if not qdrant_url and not meta_file.is_file():
            raise FileNotFoundError(f"No Qdrant index metadata found in {index_dir}. Run `sanad ingest` first.")

        meta: dict[str, Any] = {}
        if meta_file.is_file():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if meta.get("backend") and meta["backend"] != "qdrant":
                raise FileNotFoundError(f"Index in {index_dir} uses backend '{meta['backend']}', not 'qdrant'.")
            if expected_embedder and meta.get("embedder") != expected_embedder:
                raise ValueError(
                    f"Index was built with '{meta.get('embedder')}' but the current embedder is "
                    f"'{expected_embedder}'. Re-run `sanad ingest`."
                )

        resolved_path = Path(qdrant_path) if qdrant_path is not None else (index_dir / "qdrant")
        if not qdrant_url and not resolved_path.exists():
            raise FileNotFoundError(f"Qdrant storage path not found at {resolved_path}. Run `sanad ingest` first.")

        target_collection = meta.get("collection") or collection_name
        client = _build_client(qdrant_url=qdrant_url, qdrant_path=resolved_path)
        try:
            if not client.collection_exists(target_collection):
                raise FileNotFoundError(f"Qdrant collection '{target_collection}' does not exist.")

            points: list[models.Record] = []
            next_offset: Any = None
            while True:
                batch, next_offset = client.scroll(
                    collection_name=target_collection,
                    limit=256,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False,
                )
                points.extend(batch)
                if next_offset is None:
                    break

            if not points:
                raise ValueError(f"Qdrant collection '{target_collection}' is empty.")

            points.sort(key=lambda p: int(p.payload.get("chunk_index", 0) if p.payload else 0))
            stored_embedder = meta.get("embedder") or (
                points[0].payload.get("embedder", "") if points[0].payload else ""
            )
            if expected_embedder and stored_embedder and stored_embedder != expected_embedder:
                raise ValueError(
                    f"Index was built with '{stored_embedder}' but the current embedder is "
                    f"'{expected_embedder}'. Re-run `sanad ingest`."
                )

            chunk_fields = ("chunk_id", "doc_id", "title", "section", "lang", "text", "url")
            chunks = [
                Chunk(**{k: (pt.payload or {})[k] for k in chunk_fields})
                for pt in points
            ]
            return cls(
                client,
                target_collection,
                chunks,
                stored_embedder or expected_embedder or "unknown",
                qdrant_url=qdrant_url,
                qdrant_path=resolved_path,
            )
        except Exception:
            client.close()
            raise

    def ping(self) -> bool:
        try:
            return bool(self.client.collection_exists(self.collection_name))
        except Exception:
            return False

    def search(
        self,
        query: str,
        embedder: Embedder,
        top_k: int = 4,
        lang: str | None = None,
        *,
        lang_filter: str | None = None,
    ) -> list[SearchResult]:
        """Hybrid dense + lexical RRF search via Qdrant ``query_points``."""
        qvec = embedder.embed([query])[0].astype(np.float32)
        q_tokens = tokenize(query)
        sparse_q = _sparse_query_vector(q_tokens, self.vocab)
        candidate_k = max(top_k * 6, 24, len(self.chunks) if len(self.chunks) <= 64 else 24)

        q_filter = (
            models.Filter(
                must=[models.FieldCondition(key="lang", match=models.MatchValue(value=lang_filter))]
            )
            if lang_filter
            else None
        )

        if sparse_q.indices:
            response = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=qvec.tolist(),
                        using=DENSE_VECTOR_NAME,
                        limit=candidate_k,
                        filter=q_filter,
                    ),
                    models.Prefetch(
                        query=sparse_q,
                        using=SPARSE_VECTOR_NAME,
                        limit=candidate_k,
                        filter=q_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=q_filter,
                limit=candidate_k,
                with_payload=True,
                with_vectors=[DENSE_VECTOR_NAME],
            )
        else:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=qvec.tolist(),
                using=DENSE_VECTOR_NAME,
                query_filter=q_filter,
                limit=candidate_k,
                with_payload=True,
                with_vectors=[DENSE_VECTOR_NAME],
            )

        bm25_scores = self.bm25.scores(q_tokens)
        scored: list[SearchResult] = []
        for pt in response.points:
            payload = pt.payload or {}
            idx = int(payload.get("chunk_index", pt.id))
            chunk = self._chunk_by_idx[idx]
            dense_vec = (
                pt.vector.get(DENSE_VECTOR_NAME)
                if isinstance(pt.vector, dict)
                else None
            )
            if dense_vec is not None:
                dense_score = float(np.dot(np.asarray(dense_vec, dtype=np.float32), qvec))
            else:
                dense_score = float(pt.score)
            lex_score = float(bm25_scores[idx]) if idx < len(bm25_scores) else 0.0
            fused = float(pt.score)
            if lang and chunk.lang == lang:
                fused += SAME_LANG_BONUS
            scored.append(SearchResult(chunk=chunk, score=dense_score, bm25=lex_score, fused=fused))

        scored.sort(key=lambda r: r.fused, reverse=True)
        return scored[:top_k]


VectorIndex = HybridIndex | QdrantVectorIndex


def build_vector_index(settings: Any, chunks: list[Chunk], embedder: Embedder) -> VectorIndex:
    """Build either QdrantVectorIndex or legacy HybridIndex according to settings."""
    if getattr(settings, "vector_backend", "qdrant") == "numpy":
        return HybridIndex.build(chunks, embedder)
    return QdrantVectorIndex.build(
        chunks,
        embedder,
        collection_name=settings.qdrant_collection,
        qdrant_url=settings.qdrant_url,
        qdrant_path=None if settings.qdrant_url else settings.effective_qdrant_path(),
    )


def load_vector_index(settings: Any, expected_embedder: str | None = None) -> VectorIndex:
    """Load persisted Qdrant or NumPy index according to settings."""
    if getattr(settings, "vector_backend", "qdrant") == "numpy":
        return HybridIndex.load(settings.index_dir, expected_embedder=expected_embedder)
    return QdrantVectorIndex.load(
        settings.index_dir,
        expected_embedder=expected_embedder,
        collection_name=settings.qdrant_collection,
        qdrant_url=settings.qdrant_url,
        qdrant_path=None if settings.qdrant_url else settings.effective_qdrant_path(),
    )


def save_vector_index(index: VectorIndex, index_dir: Path) -> None:
    index.save(index_dir)


def load_or_build_vector_index(settings: Any, embedder: Embedder) -> VectorIndex:
    """Load the configured vector index or build and persist it from the KB."""
    try:
        return load_vector_index(settings, expected_embedder=embedder.name)
    except (FileNotFoundError, ValueError):
        index = build_vector_index(settings, build_chunks(settings.kb_dir), embedder)
        save_vector_index(index, settings.index_dir)
        return index

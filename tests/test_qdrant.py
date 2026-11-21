import numpy as np
import pytest

from sanad.config import Settings
from sanad.embeddings import HashEmbedder
from sanad.qdrant_store import (
    QdrantVectorIndex,
    build_vector_index,
    load_vector_index,
    save_vector_index,
)
from sanad.textnorm import detect_language


def test_qdrant_in_memory_hybrid_search_finds_right_section(chunks, embedder):
    q_idx = QdrantVectorIndex.build(chunks, embedder, in_memory=True)
    try:
        for query, expected_doc in [
            ("ازاي ارجع منتج", "returns-policy-ar"),
            ("how long does a card refund take?", "returns-policy-en"),
            ("مصاريف الشحن كام", "shipping-ar"),
            ("نسيت كلمة السر", "account-safety-ar"),
        ]:
            results = q_idx.search(query, embedder, top_k=3, lang=detect_language(query))
            assert results[0].chunk.doc_id == expected_doc
            assert 0.0 <= results[0].score <= 1.01
    finally:
        q_idx.close()


def test_qdrant_language_filter_and_preference(chunks, embedder):
    q_idx = QdrantVectorIndex.build(chunks, embedder, in_memory=True)
    try:
        en_only = q_idx.search("refund", embedder, top_k=4, lang="en", lang_filter="en")
        assert en_only and all(r.chunk.lang == "en" for r in en_only)

        ar_only = q_idx.search("مصاريف الشحن", embedder, top_k=4, lang="ar", lang_filter="ar")
        assert ar_only and all(r.chunk.lang == "ar" for r in ar_only)
    finally:
        q_idx.close()


def test_qdrant_disk_roundtrip_and_embedder_validation(tmp_path, chunks, embedder):
    settings = Settings()
    settings.vector_backend = "qdrant"
    settings.index_dir = tmp_path / "index"

    idx = build_vector_index(settings, chunks, embedder)
    save_vector_index(idx, settings.index_dir)
    idx.close()

    loaded = load_vector_index(settings, expected_embedder=embedder.name)
    try:
        assert isinstance(loaded, QdrantVectorIndex)
        assert loaded.ping() is True
        assert len(loaded.chunks) == len(chunks)
        hits = loaded.search("مصاريف الشحن كام", embedder, top_k=2, lang="ar")
        assert hits[0].chunk.doc_id == "shipping-ar"
    finally:
        loaded.close()

    with pytest.raises(ValueError, match="Re-run `sanad ingest`"):
        load_vector_index(settings, expected_embedder="ollama:bge-m3")


def test_qdrant_off_topic_scores_lower_than_in_domain(chunks):
    emb = HashEmbedder()
    q_idx = QdrantVectorIndex.build(chunks, emb, in_memory=True)
    try:
        on_score = q_idx.search("مصاريف الشحن كام", emb, top_k=1, lang="ar")[0].score
        off_score = q_idx.search("what is the capital of France", emb, top_k=1, lang="en")[0].score
        assert np.isfinite(on_score) and on_score > off_score
    finally:
        q_idx.close()

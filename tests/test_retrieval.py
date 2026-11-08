import httpx
import numpy as np
import pytest

from sanad.bm25 import BM25
from sanad.embeddings import HashEmbedder, OllamaEmbedder
from sanad.index import HybridIndex


def test_bm25_prefers_matching_doc():
    bm = BM25([["refund", "card"], ["shipping", "fee"], ["refund", "wallet", "instant"]])
    scores = bm.scores(["shipping"])
    assert scores[1] > 0 and scores[0] == 0 and scores[2] == 0


def test_hash_embedder_is_normalized_and_deterministic():
    emb = HashEmbedder()
    a, b = emb.embed(["مصاريف الشحن", "مصاريف الشحن"])
    assert np.isclose(np.linalg.norm(a), 1.0)
    assert np.allclose(a, b)


@pytest.mark.parametrize(
    "query, expected_doc, expected_section",
    [
        ("ازاي ارجع منتج", "returns-policy-ar", "إزاي أرجّع منتج"),
        ("how long does a card refund take?", "returns-policy-en", "Refund timing"),
        ("مصاريف الشحن كام", "shipping-ar", "مصاريف الشحن"),
        ("my listing was rejected, why?", "listing-rules-en", "Why a listing gets rejected"),
        ("نسيت كلمة السر", "account-safety-ar", "تغيير كلمة السر"),
    ],
)
def test_hybrid_search_finds_right_section(index, embedder, query, expected_doc, expected_section):
    from sanad.textnorm import detect_language

    results = index.search(query, embedder, top_k=3, lang=detect_language(query))
    top_sections = [(r.chunk.doc_id, r.chunk.section) for r in results]
    assert (expected_doc, expected_section) in top_sections
    assert results[0].chunk.doc_id == expected_doc


def test_same_language_preferred(index, embedder):
    results = index.search("refund", embedder, top_k=3, lang="en")
    assert all(r.chunk.lang == "en" for r in results)


def test_off_topic_scores_low(index, embedder):
    on = index.search("مصاريف الشحن كام", embedder, top_k=1, lang="ar")[0].score
    off = index.search("what is the capital of France", embedder, top_k=1, lang="en")[0].score
    assert on > off


def test_index_roundtrip(tmp_path, index, embedder):
    index.save(tmp_path)
    loaded = HybridIndex.load(tmp_path, expected_embedder=embedder.name)
    assert len(loaded.chunks) == len(index.chunks)
    assert np.allclose(loaded.vectors, index.vectors)
    with pytest.raises(ValueError):
        HybridIndex.load(tmp_path, expected_embedder="ollama:bge-m3")


def test_ollama_embedder_request_shape():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        assert request.url.path == "/api/embed"
        return httpx.Response(200, json={"embeddings": [[3.0, 4.0], [0.0, 2.0]]})

    client = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))
    vecs = OllamaEmbedder("http://ollama", "bge-m3", client=client).embed(["a", "b"])
    assert seen == {"model": "bge-m3", "input": ["a", "b"]}
    assert np.allclose(vecs[0], [0.6, 0.8]) and np.allclose(vecs[1], [0.0, 1.0])


def test_ollama_embedder_error_is_helpful():
    client = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    with pytest.raises(RuntimeError, match="ollama pull bge-m3"):
        OllamaEmbedder("http://ollama", "bge-m3", client=client).embed(["a"])

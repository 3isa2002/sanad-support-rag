import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from mock_backend.app import create_app as create_backend
from scripts.benchmark_vllm import benchmark_level, format_markdown
from sanad.api import create_app
from sanad.assistant import Assistant
from sanad.backend import BackendClient
from sanad.config import Settings
from sanad.llm import ExtractiveLLM, LLMError, OpenAICompatChat, make_llm
from sanad.tools import BackendTools
from sanad.tracing import AgentTracer

from .conftest import KB_DIR


def test_openai_compat_vllm_chat_and_ping():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            assert request.headers["authorization"] == "Bearer test-secret"
            return httpx.Response(200, json={"data": [{"id": "sanad-qwen25-ft"}]})
        if request.url.path == "/v1/chat/completions":
            seen.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "الشحن بـ 35 جنيه [1]."}}]},
            )
        return httpx.Response(404)

    client = httpx.Client(
        base_url="http://vllm.local/v1/",
        headers={"Authorization": "Bearer test-secret"},
        transport=httpx.MockTransport(handler),
    )
    llm = OpenAICompatChat("http://vllm.local/v1", "sanad-qwen25-ft", api_key="test-secret", client=client)
    assert llm.name == "vllm:sanad-qwen25-ft"
    assert llm.ping() is True
    reply = llm.chat([{"role": "user", "content": "مصاريف الشحن كام؟"}])
    assert reply == "الشحن بـ 35 جنيه [1]."
    assert seen["model"] == "sanad-qwen25-ft"


def test_openai_compat_vllm_error_raises_llm_error():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = httpx.Client(base_url="http://vllm.local/v1/", transport=httpx.MockTransport(boom))
    llm = OpenAICompatChat("http://vllm.local/v1", "sanad-qwen25-ft", client=client)
    assert llm.ping() is False
    with pytest.raises(LLMError, match="vLLM chat request failed"):
        llm.chat([{"role": "user", "content": "hi"}])


def test_make_llm_vllm_from_settings(monkeypatch):
    monkeypatch.setenv("SANAD_LLM", "vllm")
    monkeypatch.setenv("SANAD_VLLM_BASE_URL", "http://localhost:9000/v1")
    monkeypatch.setenv("SANAD_VLLM_MODEL", "custom-lora")
    s = Settings()
    llm = make_llm(s)
    assert isinstance(llm, OpenAICompatChat)
    assert llm.name == "vllm:custom-lora"


def test_agent_tracer_records_jsonl_and_metrics(tmp_path, index, embedder, handoffs):
    trace_file = tmp_path / "traces.jsonl"
    tracer = AgentTracer("jsonl", trace_path=trace_file)
    assistant = Assistant(index, embedder, ExtractiveLLM(), handoffs, min_score=0.15, tracer=tracer)

    r1 = assistant.answer("مصاريف الشحن كام؟", conversation_id="t-1")
    r2 = assistant.answer("what is the capital of France", conversation_id="t-2")
    assert r1.handoff is None and r2.handoff is not None

    recent = tracer.recent(limit=10)
    assert len(recent) == 2
    assert recent[0]["handed_off"] is True
    assert recent[1]["cited_count"] >= 1

    metrics = tracer.summary_metrics()
    assert metrics["total_turns"] == 2
    assert metrics["handoff_rate"] == 0.5
    assert metrics["p50_latency_ms"] >= 0.0
    assert metrics["p95_latency_ms"] >= metrics["p50_latency_ms"]


def test_api_exposes_traces_and_metrics_with_auth(tmp_path, index, embedder, handoffs):
    s = Settings()
    s.kb_dir, s.index_dir, s.db_path = KB_DIR, tmp_path / "index", tmp_path / "sanad.db"
    s.embedder, s.llm, s.admin_token = "hash", "fake", "admin-secret"
    s.tracing, s.trace_path = "jsonl", tmp_path / "traces.jsonl"

    tracer = AgentTracer.from_settings(s)
    tools = BackendTools(BackendClient("http://testserver", client=TestClient(create_backend(api_key=""))))
    assistant = Assistant(index, embedder, ExtractiveLLM(), handoffs, min_score=0.15, tools=tools, tracer=tracer)
    client = TestClient(create_app(s, assistant))

    assert client.post("/chat", json={"message": "مصاريف الشحن كام؟"}).status_code == 200
    assert client.get("/traces").status_code == 401
    traces = client.get("/traces", headers={"X-Admin-Token": "admin-secret"}).json()
    assert len(traces) == 1 and traces[0]["language"] == "ar"

    metrics = client.get("/metrics").json()
    assert metrics["total_turns"] == 1
    assert client.get("/health").json()["tracing"] == "jsonl"


def test_benchmark_vllm_sse_streaming_parser(monkeypatch):
    sse_body = (
        'data: {"choices":[{"delta":{"content":"الشحن "}}]}\n'
        'data: {"choices":[{"delta":{"content":"35 جنيه [1]."}}],"usage":{"completion_tokens":8}}\n'
        "data: [DONE]\n"
    ).encode("utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse_body, headers={"content-type": "text/event-stream"})

    orig_async_client = httpx.AsyncClient

    def patched_async_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return orig_async_client(*args, **kwargs)

    monkeypatch.setattr("scripts.benchmark_vllm.httpx.AsyncClient", patched_async_client)
    res = asyncio.run(
        benchmark_level(
            "http://vllm.local/v1",
            "sanad-qwen25-ft",
            api_key="EMPTY",
            concurrency=2,
            num_requests=4,
            max_tokens=64,
            timeout=10.0,
        )
    )
    assert res.succeeded == 4 and res.failed == 0
    assert res.output_tokens_total == 32
    assert res.output_tokens_per_sec > 0
    md = format_markdown({"base_url": "http://vllm.local/v1", "model": "sanad-qwen25-ft", "gpu": None, "results": [res.__dict__]})
    assert "vLLM Serving Benchmark Report" in md and "sanad-qwen25-ft" in md


def test_api_cors_and_private_network_preflight(tmp_path, index, embedder, handoffs):
    s = Settings()
    s.kb_dir, s.index_dir, s.db_path = KB_DIR, tmp_path / "index", tmp_path / "sanad.db"
    s.embedder, s.llm = "hash", "fake"
    assistant = Assistant(index, embedder, ExtractiveLLM(), handoffs, min_score=0.15)
    client = TestClient(create_app(s, assistant))

    preflight = client.options(
        "/chat",
        headers={
            "Origin": "https://mohamed-eissa.pages.dev",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Private-Network": "true",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers.get("access-control-allow-origin") in ("*", "https://mohamed-eissa.pages.dev")
    assert preflight.headers.get("access-control-allow-private-network") == "true"


def test_qdrant_build_recovers_from_corrupt_local_collection(tmp_path, embedder):
    from sanad.kb import build_chunks
    from sanad.qdrant_store import QdrantVectorIndex

    qdrant_dir = tmp_path / "index" / "qdrant"
    bad_col = qdrant_dir / "collection" / "sanad_kb"
    bad_col.mkdir(parents=True)
    # Create an invalid local Qdrant state that raises ValueError on QdrantClient(path=...)
    (qdrant_dir / "meta.json").write_text('{"collections":{"sanad_kb":{"vectors":{"dense":{"size":1024,"distance":"Cosine"}}}}}', encoding="utf-8")
    (bad_col / "storage.sqlite").write_bytes(b"corrupt")

    idx = QdrantVectorIndex.build(build_chunks(KB_DIR), embedder, qdrant_path=qdrant_dir)
    assert len(idx.chunks) > 0
    idx.close()

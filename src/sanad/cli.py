"""Command-line interface: sanad ingest | search | ask | serve | eval | ragas | graph."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Settings
from .embeddings import make_embedder
from .kb import build_chunks
from .qdrant_store import build_vector_index, load_vector_index, save_vector_index


def _close_index(index: object) -> None:
    if hasattr(index, "close"):
        index.close()


def cmd_ingest(settings: Settings, args: argparse.Namespace) -> int:
    chunks = build_chunks(settings.kb_dir)
    embedder = make_embedder(settings)
    index = build_vector_index(settings, chunks, embedder)
    try:
        save_vector_index(index, settings.index_dir)
        docs = len({c.doc_id for c in chunks})
        print(
            f"Indexed {len(chunks)} chunks from {docs} documents with {embedder.name} "
            f"({settings.vector_backend}) → {settings.index_dir}"
        )
    finally:
        _close_index(index)
    return 0


def cmd_search(settings: Settings, args: argparse.Namespace) -> int:
    from .textnorm import detect_language

    embedder = make_embedder(settings)
    index = load_vector_index(settings, expected_embedder=embedder.name)
    try:
        results = index.search(args.query, embedder, top_k=args.k, lang=detect_language(args.query))
        for i, r in enumerate(results, 1):
            snippet = r.chunk.text.replace("\n", " ")[:110]
            print(f"[{i}] {r.chunk.doc_id} › {r.chunk.section}  (cos={r.score:.3f} bm25={r.bm25:.2f})\n    {snippet}")
    finally:
        _close_index(index)
    return 0


def cmd_ask(settings: Settings, args: argparse.Namespace) -> int:
    from .factory import build_assistant

    assistant = build_assistant(settings)
    try:
        response = assistant.answer(args.question, conversation_id=args.conversation_id)
        if args.json:
            print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))
            return 0
        print(response.answer)
        for i, s in enumerate(response.sources, 1):
            print(f"  [{i}] {s.title} › {s.section}  {s.url}")
        if response.handoff:
            print(f"  → handoff {response.handoff.ticket_id} ({response.handoff.reason})")
        if response.trace:
            print(f"  → trace {' -> '.join(response.trace)}")
    finally:
        _close_index(assistant.index)
    return 0


def cmd_serve(settings: Settings, args: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(settings), host=args.host or settings.host, port=args.port or settings.port)
    return 0


def _make_eval_tools(settings: Settings, *, live_backend: bool):
    from .tools import BackendTools

    if live_backend:
        from .factory import make_tools

        return make_tools(settings)

    import warnings

    warnings.filterwarnings("ignore", message="Using `httpx` with")
    from fastapi.testclient import TestClient

    from mock_backend.app import create_app
    from .backend import BackendClient

    return BackendTools(BackendClient("http://testserver", client=TestClient(create_app(api_key=""))))


def cmd_eval(settings: Settings, args: argparse.Namespace) -> int:
    from .assistant import Assistant
    from .eval import evaluate, load_questions
    from .handoff import HandoffStore
    from .llm import make_llm

    embedder = make_embedder(settings)
    index = load_vector_index(settings, expected_embedder=embedder.name)
    try:
        tools = _make_eval_tools(settings, live_backend=args.live_backend)
        min_score = args.min_score if args.min_score is not None else settings.effective_min_score(embedder.name)
        assistant = Assistant(
            index,
            embedder,
            make_llm(settings),
            HandoffStore(":memory:"),
            top_k=settings.top_k,
            min_score=min_score,
            tools=tools,
            grounding_retries=settings.grounding_retries,
            strict_grounding=settings.strict_grounding,
            llm_grounding_check=settings.llm_grounding_check,
            prompt_style=settings.effective_prompt_style(),
        )
        report = evaluate(assistant, load_questions(args.file))
        summary = report.summary()
        print(
            f"backend={settings.vector_backend} embedder={embedder.name} "
            f"llm={assistant.llm.name} min_score={min_score} top_k={settings.top_k}"
        )
        for key, value in summary.items():
            print(f"  {key:<24} {value}")
        if args.verbose and report.failures:
            print("failures:")
            for f in report.failures:
                print("  -", json.dumps(f, ensure_ascii=False))
    finally:
        _close_index(index)
    return 0


def cmd_ragas(settings: Settings, args: argparse.Namespace) -> int:
    from .assistant import Assistant
    from .eval import load_questions
    from .handoff import HandoffStore
    from .llm import make_llm
    from .ragas_eval import (
        OllamaStructuredRagasLLM,
        SanadRagasEmbedding,
        run_ragas_benchmark,
        save_ragas_report,
    )

    embedder = make_embedder(settings)
    index = load_vector_index(settings, expected_embedder=embedder.name)
    try:
        tools = _make_eval_tools(settings, live_backend=False)
        min_score = args.min_score if args.min_score is not None else settings.effective_min_score(embedder.name)
        assistant = Assistant(
            index,
            embedder,
            make_llm(settings),
            HandoffStore(":memory:"),
            top_k=settings.top_k,
            min_score=min_score,
            tools=tools,
            grounding_retries=settings.grounding_retries,
            strict_grounding=settings.strict_grounding,
            prompt_style=settings.effective_prompt_style(),
        )
        judge_model = args.judge_model or settings.ragas_model
        ragas_llm = OllamaStructuredRagasLLM(settings.ollama_base_url, judge_model, timeout=settings.llm_timeout)
        ragas_emb = SanadRagasEmbedding(embedder)
        report, _ = run_ragas_benchmark(
            assistant,
            load_questions(args.file),
            ragas_llm=ragas_llm,
            ragas_embeddings=ragas_emb,
            vector_backend=settings.vector_backend,
            judge_name=f"ollama:{judge_model}",
            limit=args.limit,
            strictness=args.strictness,
        )
        out_json = Path(args.output)
        out_md = out_json.with_suffix(".md")
        save_ragas_report(report, out_json, out_md)
        print(report.to_markdown())
        print(f"Saved RAGAS report → {out_json} and {out_md}")
    finally:
        _close_index(index)
    return 0


def cmd_graph(settings: Settings, args: argparse.Namespace) -> int:
    from .assistant import Assistant
    from .handoff import HandoffStore
    from .index import HybridIndex
    from .embeddings import HashEmbedder
    from .llm import ExtractiveLLM
    from .kb import Chunk

    dummy_index = HybridIndex.build(
        [Chunk(chunk_id="c0", doc_id="d0", title="t", section="s", lang="en", text="placeholder")],
        HashEmbedder(),
    )
    assistant = Assistant(dummy_index, HashEmbedder(), ExtractiveLLM(), HandoffStore(":memory:"))
    print(assistant.graph_mermaid())
    return 0


def cmd_traces(settings: Settings, args: argparse.Namespace) -> int:
    from .tracing import AgentTracer

    tracer = AgentTracer.from_settings(settings)
    print(json.dumps({"summary": tracer.summary_metrics(), "recent": tracer.recent(limit=args.limit)}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sanad", description="Sanad — bilingual multi-agent support assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Chunk and index the knowledge base")

    p = sub.add_parser("search", help="Show the top retrieved chunks for a query")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=4)

    p = sub.add_parser("ask", help="Ask Sanad a question")
    p.add_argument("question")
    p.add_argument("--conversation-id", default=None, help="Thread ID for multi-turn state")
    p.add_argument("--json", action="store_true", help="Print the full JSON response")

    p = sub.add_parser("serve", help="Run the HTTP API and web chat")
    p.add_argument("--host")
    p.add_argument("--port", type=int)

    p = sub.add_parser("eval", help="Evaluate on the golden question set")
    p.add_argument("--file", default="data/eval/questions.jsonl")
    p.add_argument("--min-score", type=float, default=None, help="Override the handoff threshold")
    p.add_argument("--live-backend", action="store_true", help="Use SANAD_BACKEND_URL instead of the in-process mock")
    p.add_argument("-v", "--verbose", action="store_true", help="List failures")

    p = sub.add_parser("ragas", help="Run RAGAS Faithfulness & AnswerRelevancy benchmark")
    p.add_argument("--file", default="data/eval/questions.jsonl")
    p.add_argument("--output", default="data/eval/ragas_report.json")
    p.add_argument("--judge-model", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--strictness", type=int, default=1)
    p.add_argument("--min-score", type=float, default=None)

    sub.add_parser("graph", help="Print the Mermaid diagram of Sanad's LangGraph workflow")

    p = sub.add_parser("traces", help="Inspect agent telemetry summary and recent traces")
    p.add_argument("--limit", type=int, default=10)

    return parser


COMMANDS = {
    "ingest": cmd_ingest,
    "search": cmd_search,
    "ask": cmd_ask,
    "serve": cmd_serve,
    "eval": cmd_eval,
    "ragas": cmd_ragas,
    "graph": cmd_graph,
    "traces": cmd_traces,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    try:
        return COMMANDS[args.command](settings, args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

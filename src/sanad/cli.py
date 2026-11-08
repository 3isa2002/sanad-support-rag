"""Command-line interface: sanad ingest | search | ask | serve | eval."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import Settings
from .embeddings import make_embedder
from .index import HybridIndex
from .kb import build_chunks


def cmd_ingest(settings: Settings, args: argparse.Namespace) -> int:
    chunks = build_chunks(settings.kb_dir)
    embedder = make_embedder(settings)
    index = HybridIndex.build(chunks, embedder)
    index.save(settings.index_dir)
    docs = len({c.doc_id for c in chunks})
    print(f"Indexed {len(chunks)} chunks from {docs} documents with {embedder.name} → {settings.index_dir}")
    return 0


def cmd_search(settings: Settings, args: argparse.Namespace) -> int:
    from .textnorm import detect_language

    embedder = make_embedder(settings)
    index = HybridIndex.load(settings.index_dir, expected_embedder=embedder.name)
    results = index.search(args.query, embedder, top_k=args.k, lang=detect_language(args.query))
    for i, r in enumerate(results, 1):
        snippet = r.chunk.text.replace("\n", " ")[:110]
        print(f"[{i}] {r.chunk.doc_id} › {r.chunk.section}  (cos={r.score:.3f} bm25={r.bm25:.2f})\n    {snippet}")
    return 0


def cmd_ask(settings: Settings, args: argparse.Namespace) -> int:
    from .factory import build_assistant

    response = build_assistant(settings).answer(args.question)
    if args.json:
        print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))
        return 0
    print(response.answer)
    for i, s in enumerate(response.sources, 1):
        print(f"  [{i}] {s.title} › {s.section}  {s.url}")
    if response.handoff:
        print(f"  → handoff {response.handoff.ticket_id} ({response.handoff.reason})")
    return 0


def cmd_serve(settings: Settings, args: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(settings), host=args.host or settings.host, port=args.port or settings.port)
    return 0


def cmd_eval(settings: Settings, args: argparse.Namespace) -> int:
    from .assistant import Assistant
    from .eval import evaluate, load_questions
    from .handoff import HandoffStore
    from .llm import make_llm
    from .tools import BackendTools

    embedder = make_embedder(settings)
    index = HybridIndex.load(settings.index_dir, expected_embedder=embedder.name)
    if args.live_backend:
        from .factory import make_tools

        tools = make_tools(settings)
    else:  # in-process mock backend, so eval needs no running services
        import warnings

        warnings.filterwarnings("ignore", message="Using `httpx` with")
        from fastapi.testclient import TestClient

        from mock_backend.app import create_app
        from .backend import BackendClient

        tools = BackendTools(BackendClient("http://testserver", client=TestClient(create_app(api_key=""))))
    min_score = args.min_score if args.min_score is not None else settings.effective_min_score(embedder.name)
    assistant = Assistant(
        index, embedder, make_llm(settings), HandoffStore(":memory:"),
        top_k=settings.top_k, min_score=min_score, tools=tools,
    )
    report = evaluate(assistant, load_questions(args.file))
    summary = report.summary()
    print(f"embedder={embedder.name} llm={assistant.llm.name} min_score={min_score} top_k={settings.top_k}")
    for key, value in summary.items():
        print(f"  {key:<24} {value}")
    if args.verbose and report.failures:
        print("failures:")
        for f in report.failures:
            print("  -", json.dumps(f, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sanad", description="Sanad — bilingual support assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Chunk and index the knowledge base")

    p = sub.add_parser("search", help="Show the top retrieved chunks for a query")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=4)

    p = sub.add_parser("ask", help="Ask Sanad a question")
    p.add_argument("question")
    p.add_argument("--json", action="store_true", help="Print the full JSON response")

    p = sub.add_parser("serve", help="Run the HTTP API and web chat")
    p.add_argument("--host")
    p.add_argument("--port", type=int)

    p = sub.add_parser("eval", help="Evaluate on the golden question set")
    p.add_argument("--file", default="data/eval/questions.jsonl")
    p.add_argument("--min-score", type=float, default=None, help="Override the handoff threshold")
    p.add_argument("--live-backend", action="store_true", help="Use SANAD_BACKEND_URL instead of the in-process mock")
    p.add_argument("-v", "--verbose", action="store_true", help="List failures")

    return parser


COMMANDS = {
    "ingest": cmd_ingest,
    "search": cmd_search,
    "ask": cmd_ask,
    "serve": cmd_serve,
    "eval": cmd_eval,
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

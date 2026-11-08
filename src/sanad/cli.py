"""Command-line interface: sanad ingest | search | ask | serve | eval."""

from __future__ import annotations

import argparse
import json
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

    return parser


COMMANDS = {
    "ingest": cmd_ingest,
    "search": cmd_search,
    "ask": cmd_ask,
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

"""Wire the assistant together from settings."""

from __future__ import annotations

from .assistant import Assistant
from .config import Settings
from .embeddings import make_embedder
from .handoff import HandoffStore
from .index import HybridIndex
from .llm import make_llm


def build_assistant(settings: Settings, *, index: HybridIndex | None = None, tools=None) -> Assistant:
    embedder = make_embedder(settings)
    if index is None:
        index = HybridIndex.load(settings.index_dir, expected_embedder=embedder.name)
    return Assistant(
        index=index,
        embedder=embedder,
        llm=make_llm(settings),
        handoffs=HandoffStore(settings.db_path),
        top_k=settings.top_k,
        min_score=settings.effective_min_score(embedder.name),
        tools=tools,
    )

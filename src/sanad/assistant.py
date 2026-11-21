"""Orchestration facade backed by Sanad's stateful multi-agent LangGraph workflow."""

from __future__ import annotations

from typing import Any

from .embeddings import Embedder
from .graph import build_sanad_graph, invoke_sanad_graph
from .handoff import HandoffStore
from .llm import ChatLLM
from .messages import msg
from .models import (
    AssistantResponse,
    Handoff,
    Source,
    ToolCall,
    is_handoff_reply,
    parse_citations,
    source_from_result,
    wants_human,
)
from .qdrant_store import VectorIndex

__all__ = [
    "Assistant",
    "AssistantResponse",
    "Handoff",
    "Source",
    "ToolCall",
    "is_handoff_reply",
    "parse_citations",
    "wants_human",
]


class Assistant:
    """Multi-agent customer-support assistant powered by LangGraph and Qdrant/Hybrid retrieval."""

    def __init__(
        self,
        index: VectorIndex,
        embedder: Embedder,
        llm: ChatLLM,
        handoffs: HandoffStore,
        *,
        top_k: int = 4,
        min_score: float = 0.45,
        tools: Any | None = None,
        grounding_retries: int = 0,
        strict_grounding: bool = False,
        llm_grounding_check: bool = False,
    ):
        self.index = index
        self.embedder = embedder
        self.llm = llm
        self.handoffs = handoffs
        self.top_k = top_k
        self.min_score = min_score
        self.tools = tools
        self.grounding_retries = max(0, grounding_retries)
        self.strict_grounding = strict_grounding
        self.llm_grounding_check = llm_grounding_check
        self._graph = build_sanad_graph(self)

    def answer(self, question: str, conversation_id: str | None = None) -> AssistantResponse:
        return invoke_sanad_graph(
            self._graph,
            question,
            conversation_id=conversation_id,
            max_attempts=1 + self.grounding_retries,
        )

    def graph_mermaid(self) -> str:
        """Return a Mermaid diagram of the compiled LangGraph workflow."""
        return self._graph.get_graph().draw_mermaid()

    def _handoff(
        self,
        question: str,
        lang: str,
        reason: str,
        conversation_id: str | None,
        *,
        context: dict | None = None,
        key: str = "handoff",
        tool_calls: list[ToolCall] | None = None,
        **fmt: str,
    ) -> AssistantResponse:
        ticket = self.handoffs.create(question, lang, reason, conversation_id, context)
        return AssistantResponse(
            answer=msg(key, lang, ticket_id=ticket.id, **fmt),
            language=lang,
            handoff=Handoff(ticket_id=ticket.id, reason=reason),
            tool_calls=tool_calls or [],
        )

"""Orchestration: route a customer message to handoff, backend tools or grounded RAG."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from .embeddings import Embedder
from .handoff import HandoffStore
from .index import HybridIndex, SearchResult
from .llm import ChatLLM, LLMError
from .messages import msg
from .prompts import HANDOFF_MARKER, grounded_messages
from .textnorm import detect_language, normalize, to_latin_digits


class Source(BaseModel):
    doc_id: str
    title: str
    section: str
    url: str = ""
    score: float
    snippet: str


class Handoff(BaseModel):
    ticket_id: str
    reason: str


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]
    ok: bool
    result: dict[str, Any] | None = None
    error: str | None = None


class AssistantResponse(BaseModel):
    answer: str
    language: str
    sources: list[Source] = Field(default_factory=list)
    handoff: Handoff | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


# ---------------------------------------------------------------- helpers

_HUMAN_PATTERNS = [
    # Arabic (normalized: ة→ه, أ/إ→ا, ى→ي)
    r"(اكلم|اتكلم مع|كلمني|عايز|عاوز|محتاج|اريد|ابغي|حولني|وصلني)\s*(ل|لـ|مع|علي)?\s*(حد|موظف|انسان|بني ادم|خدمه العملاء|الدعم)",
    r"\b(موظف|خدمه العملاء)\b.*\b(حقيقي|بشري)\b",
    r"^(موظف|خدمه العملاء|كلم موظف)[\s؟?!.]*$",
    # English
    r"\b(talk|speak|chat)\s+(to|with)\s+(a\s+|an\s+)?(human|person|agent|representative|someone|real person|support)\b",
    r"\b(human|live|real)\s+(agent|person|support)\b",
    r"^\s*(agent|human|representative)[\s?!.]*$",
]
_HUMAN_RE = [re.compile(p) for p in _HUMAN_PATTERNS]
_CITATION = re.compile(r"\[([\d\s,،]+)\]")


def wants_human(text: str) -> bool:
    norm = normalize(text)
    return any(p.search(norm) for p in _HUMAN_RE)


def parse_citations(answer: str, n_passages: int) -> list[int]:
    """Return the distinct, valid 1-based passage numbers cited in ``answer``, in order."""
    cited: list[int] = []
    for group in _CITATION.findall(to_latin_digits(answer)):
        for part in re.split(r"[\s,،]+", group.strip()):
            if part.isdigit():
                n = int(part)
                if 1 <= n <= n_passages and n not in cited:
                    cited.append(n)
    return cited


def is_handoff_reply(answer: str) -> bool:
    stripped = answer.strip().strip(".!").upper()
    return stripped == HANDOFF_MARKER or bool(re.search(rf"\b{HANDOFF_MARKER}\b", answer))


def _source(result: SearchResult) -> Source:
    c = result.chunk
    snippet = c.text.replace("\n", " ")
    return Source(
        doc_id=c.doc_id,
        title=c.title,
        section=c.section,
        url=c.url,
        score=round(result.score, 4),
        snippet=snippet[:240] + ("…" if len(snippet) > 240 else ""),
    )


# ---------------------------------------------------------------- assistant


class Assistant:
    def __init__(
        self,
        index: HybridIndex,
        embedder: Embedder,
        llm: ChatLLM,
        handoffs: HandoffStore,
        *,
        top_k: int = 4,
        min_score: float = 0.45,
        tools: Any | None = None,
    ):
        self.index = index
        self.embedder = embedder
        self.llm = llm
        self.handoffs = handoffs
        self.top_k = top_k
        self.min_score = min_score
        self.tools = tools

    def answer(self, question: str, conversation_id: str | None = None) -> AssistantResponse:
        question = question.strip()
        lang = detect_language(question)
        if not question:
            return AssistantResponse(answer=msg("empty", lang), language=lang)

        if wants_human(question):
            return self._handoff(question, lang, "human_requested", conversation_id, key="human_requested")

        if self.tools is not None:
            tool_response = self.tools.handle(question, lang, self, conversation_id)
            if tool_response is not None:
                return tool_response

        return self._rag(question, lang, conversation_id)

    # ------------------------------------------------------------ RAG path
    def _rag(self, question: str, lang: str, conversation_id: str | None) -> AssistantResponse:
        results = self.index.search(question, self.embedder, top_k=self.top_k, lang=lang)
        best = max((r.score for r in results), default=0.0)
        retrieved = [_source(r).model_dump() for r in results]
        if best < self.min_score:
            return self._handoff(
                question, lang, "low_confidence", conversation_id,
                context={"best_score": round(best, 4), "min_score": self.min_score, "retrieved": retrieved},
            )

        try:
            reply = self.llm.chat(grounded_messages(question, results, lang))
        except LLMError as exc:
            return self._handoff(question, lang, "llm_error", conversation_id, context={"error": str(exc)})

        if is_handoff_reply(reply):
            return self._handoff(question, lang, "model_uncertain", conversation_id, context={"retrieved": retrieved})

        cited = parse_citations(reply, len(results))
        if not cited:
            return self._handoff(
                question, lang, "no_citation", conversation_id,
                context={"draft_answer": reply, "retrieved": retrieved},
            )

        return AssistantResponse(
            answer=reply,
            language=lang,
            sources=[_source(results[n - 1]) for n in cited],
        )

    # ------------------------------------------------------------ handoff
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

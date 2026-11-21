"""Shared response models and deterministic citation/intent helpers."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from .index import SearchResult
from .textnorm import normalize, to_latin_digits

HANDOFF_MARKER = "HANDOFF"


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
    trace: list[str] = Field(default_factory=list)
    grounding_issues: list[str] = Field(default_factory=list)


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


def extract_all_citation_numbers(answer: str) -> list[int]:
    """Return every raw integer cited inside ``[n]`` brackets in ``answer``."""
    found: list[int] = []
    for group in _CITATION.findall(to_latin_digits(answer)):
        for part in re.split(r"[\s,،]+", group.strip()):
            if part.isdigit():
                found.append(int(part))
    return found


def parse_citations(answer: str, n_passages: int) -> list[int]:
    """Return the distinct, valid 1-based passage numbers cited in ``answer``, in order."""
    cited: list[int] = []
    for n in extract_all_citation_numbers(answer):
        if 1 <= n <= n_passages and n not in cited:
            cited.append(n)
    return cited


def is_handoff_reply(answer: str) -> bool:
    stripped = answer.strip().strip(".!").upper()
    return stripped == HANDOFF_MARKER or bool(re.search(rf"\b{HANDOFF_MARKER}\b", answer))


def source_from_result(result: SearchResult) -> Source:
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

"""Offline evaluation over a golden question set.

Each JSONL line: {"question": str, "expect": "answer"|"handoff"|"tool", "doc"?: str, "tool"?: str}

Metrics
- retrieval hit@k   — for "answer" items, is the expected document among the top-k chunks?
- answer rate        — "answer" items that were answered (not handed off) with ≥1 citation
- handoff precision / recall — treating "handoff" items as positives
- tool routing accuracy      — "tool" items routed to the expected backend tool
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .assistant import Assistant


@dataclass
class EvalReport:
    total: int = 0
    retrieval_hits: int = 0
    retrieval_total: int = 0
    answered: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tool_ok: int = 0
    tool_total: int = 0
    failures: list[dict] = field(default_factory=list)

    @staticmethod
    def _ratio(a: int, b: int) -> float:
        return a / b if b else 0.0

    def summary(self) -> dict:
        return {
            "questions": self.total,
            "retrieval_hit_at_k": round(self._ratio(self.retrieval_hits, self.retrieval_total), 3),
            "answer_rate": round(self._ratio(self.answered, self.retrieval_total), 3),
            "handoff_precision": round(self._ratio(self.tp, self.tp + self.fp), 3),
            "handoff_recall": round(self._ratio(self.tp, self.tp + self.fn), 3),
            "tool_routing_accuracy": round(self._ratio(self.tool_ok, self.tool_total), 3),
        }


def load_questions(path: Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def evaluate(assistant: Assistant, questions: list[dict], k: int | None = None) -> EvalReport:
    from .textnorm import detect_language

    k = k or assistant.top_k
    report = EvalReport()
    for item in questions:
        q, expect = item["question"], item["expect"]
        report.total += 1
        res = assistant.answer(q)
        handed_off = res.handoff is not None

        if expect == "answer":
            report.retrieval_total += 1
            hits = assistant.index.search(q, assistant.embedder, top_k=k, lang=detect_language(q))
            if item["doc"] in {h.chunk.doc_id for h in hits}:
                report.retrieval_hits += 1
            else:
                report.failures.append({"question": q, "problem": "retrieval_miss", "expected": item["doc"]})
            if not handed_off and res.sources:
                report.answered += 1
            else:
                reason = res.handoff.reason if res.handoff else "no_sources"
                report.failures.append({"question": q, "problem": "not_answered", "reason": reason})

        if expect == "tool":
            report.tool_total += 1
            if any(t.name == item["tool"] and t.ok for t in res.tool_calls):
                report.tool_ok += 1
            else:
                report.failures.append({"question": q, "problem": "tool_miss", "expected": item["tool"]})

        should_handoff = expect == "handoff"
        if handed_off and should_handoff:
            report.tp += 1
        elif handed_off:
            report.fp += 1
        elif should_handoff:
            report.fn += 1
            report.failures.append({"question": q, "problem": "missed_handoff", "answer": res.answer[:120]})
    return report

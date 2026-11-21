"""Agent tracing and telemetry (JSONL spans, Langfuse, and LangSmith)."""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("sanad.tracing")


@dataclass
class TraceRecord:
    trace_id: str
    timestamp: str
    conversation_id: str | None
    question: str
    route: str
    language: str
    agent_trace: list[str]
    latency_ms: float
    cited_count: int
    handed_off: bool
    handoff_reason: str | None
    grounding_issues: list[str]
    llm_name: str
    embedder_name: str
    external_trace_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return round(sorted_vals[0], 2)
    idx = (len(sorted_vals) - 1) * pct
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return round(sorted_vals[lo], 2)
    frac = idx - lo
    return round(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac, 2)


class AgentTracer:
    """Thread-safe tracer emitting structured JSONL spans plus optional Langfuse / LangSmith hooks."""

    def __init__(
        self,
        backend: str = "jsonl",
        *,
        trace_path: Path | str = Path("data/traces.jsonl"),
        langfuse_client: Any = None,
        langfuse_handler_cls: Any = None,
    ):
        self.backend = backend
        self.trace_path = Path(trace_path)
        self._lock = threading.Lock()
        self._langfuse = langfuse_client
        self._langfuse_handler_cls = langfuse_handler_cls

    @classmethod
    def from_settings(cls, settings: Any) -> "AgentTracer":
        backend = (getattr(settings, "tracing", "jsonl") or "none").lower()
        trace_path = getattr(settings, "trace_path", Path("data/traces.jsonl"))

        if backend == "langsmith":
            os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
            os.environ.setdefault("LANGSMITH_PROJECT", getattr(settings, "langsmith_project", "sanad-support-rag"))
            return cls(backend="langsmith", trace_path=trace_path)

        if backend == "langfuse":
            pub = getattr(settings, "langfuse_public_key", "")
            sec = getattr(settings, "langfuse_secret_key", "")
            host = getattr(settings, "langfuse_base_url", "http://localhost:3000")
            if pub and sec:
                try:
                    from langfuse import Langfuse
                    from langfuse.langchain import CallbackHandler

                    lf = Langfuse(public_key=pub, secret_key=sec, host=host)
                    return cls(
                        backend="langfuse",
                        trace_path=trace_path,
                        langfuse_client=lf,
                        langfuse_handler_cls=CallbackHandler,
                    )
                except Exception as exc:
                    log.warning("Langfuse init failed (%s); falling back to JSONL tracing.", exc)
            else:
                log.warning("SANAD_TRACING=langfuse set without LANGFUSE_PUBLIC_KEY/SECRET_KEY; falling back to JSONL.")
            return cls(backend="jsonl", trace_path=trace_path)

        if backend in ("jsonl", "none"):
            return cls(backend=backend, trace_path=trace_path)

        raise ValueError(f"Unknown SANAD_TRACING: {backend!r} (use 'jsonl', 'langfuse', 'langsmith', or 'none')")

    def callbacks_for_run(self) -> tuple[list[Any], Any]:
        """Return LangGraph callback handlers and an optional Langfuse callback object."""
        if self.backend == "langfuse" and self._langfuse_handler_cls is not None:
            try:
                handler = self._langfuse_handler_cls()
                return [handler], handler
            except Exception as exc:
                log.warning("Failed to create Langfuse CallbackHandler: %s", exc)
        return [], None

    def record(
        self,
        *,
        question: str,
        response: Any,
        latency_ms: float,
        llm_name: str,
        embedder_name: str,
        conversation_id: str | None = None,
        handler: Any = None,
    ) -> TraceRecord | None:
        if self.backend == "none":
            return None

        trace = list(getattr(response, "trace", []) or [])
        route = trace[1] if len(trace) > 1 else (trace[0] if trace else "unknown")
        ext_id = getattr(handler, "last_trace_id", None)

        if self._langfuse is not None:
            try:
                with self._langfuse.start_as_current_observation(
                    as_type="span",
                    name="sanad-turn",
                    input={"question": question, "conversation_id": conversation_id},
                    metadata={
                        "route": route,
                        "language": response.language,
                        "agent_trace": trace,
                        "latency_ms": round(latency_ms, 2),
                        "handed_off": response.handoff is not None,
                        "handoff_reason": response.handoff.reason if response.handoff else None,
                    },
                ) as span:
                    span.update(output={"answer": response.answer, "sources": len(response.sources)})
                    ext_id = ext_id or getattr(span, "trace_id", None)
                self._langfuse.flush()
            except Exception as exc:
                log.warning("Langfuse span logging failed: %s", exc)

        rec = TraceRecord(
            trace_id=uuid.uuid4().hex[:12],
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            conversation_id=conversation_id,
            question=question,
            route=route,
            language=response.language,
            agent_trace=trace,
            latency_ms=round(latency_ms, 2),
            cited_count=len(response.sources),
            handed_off=response.handoff is not None,
            handoff_reason=response.handoff.reason if response.handoff else None,
            grounding_issues=list(getattr(response, "grounding_issues", []) or []),
            llm_name=llm_name,
            embedder_name=embedder_name,
            external_trace_id=ext_id,
        )

        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.trace_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        return rec

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.trace_path.is_file():
            return []
        with self._lock:
            lines = [ln for ln in self.trace_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        rows = [json.loads(ln) for ln in lines[-max(1, limit) :]]
        rows.reverse()
        return rows

    def summary_metrics(self) -> dict[str, Any]:
        rows = self.recent(limit=1000)
        if not rows:
            return {
                "tracing_backend": self.backend,
                "total_turns": 0,
                "avg_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "handoff_rate": 0.0,
                "routes": {},
            }
        latencies = sorted(float(r.get("latency_ms", 0.0)) for r in rows)
        handoffs = sum(1 for r in rows if r.get("handed_off"))
        routes: dict[str, int] = {}
        for r in rows:
            rt = str(r.get("route", "unknown"))
            routes[rt] = routes.get(rt, 0) + 1
        return {
            "tracing_backend": self.backend,
            "total_turns": len(rows),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            "p50_latency_ms": _percentile(latencies, 0.50),
            "p95_latency_ms": _percentile(latencies, 0.95),
            "handoff_rate": round(handoffs / len(rows), 4),
            "routes": routes,
        }

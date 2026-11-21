"""RAGAS evaluation pipeline (Faithfulness & AnswerRelevancy) for Sanad v2."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Type, TypeVar

import httpx
from pydantic import BaseModel
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from .assistant import Assistant
from .embeddings import Embedder
from .textnorm import detect_language

TModel = TypeVar("TModel", bound=BaseModel)
_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class SanadRagasEmbedding(BaseRagasEmbedding):
    """Bridge Sanad's ``Embedder`` (``OllamaEmbedder`` / ``HashEmbedder``) to RAGAS."""

    def __init__(self, embedder: Embedder):
        super().__init__()
        self._embedder = embedder

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        vec = self._embedder.embed([text])[0]
        return [float(x) for x in vec.tolist()]

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return self.embed_text(text, **kwargs)


def _extract_json_payload(raw: str) -> str:
    stripped = raw.strip()
    match = _JSON_FENCE.search(stripped)
    if match:
        return match.group(1).strip()
    return stripped


class OllamaStructuredRagasLLM(InstructorBaseRagasLLM):
    """Structured-output RAGAS judge backed by Ollama's native JSON schema decoding."""

    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _payload(self, prompt: str, response_model: Type[TModel]) -> dict[str, Any]:
        schema = response_model.model_json_schema()
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Respond strictly with valid JSON matching the requested schema.",
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": schema,
            "options": {"temperature": 0.0},
        }

    def generate(self, prompt: str, response_model: Type[TModel]) -> TModel:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            resp = client.post("/api/chat", json=self._payload(prompt, response_model))
            resp.raise_for_status()
        content = resp.json()["message"]["content"]
        return response_model.model_validate_json(_extract_json_payload(content))

    async def agenerate(self, prompt: str, response_model: Type[TModel]) -> TModel:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            resp = await client.post("/api/chat", json=self._payload(prompt, response_model))
            resp.raise_for_status()
        content = resp.json()["message"]["content"]
        return response_model.model_validate_json(_extract_json_payload(content))


@dataclass
class RagasSampleScore:
    question: str
    language: str
    response: str
    retrieved_contexts: list[str]
    handed_off: bool
    faithfulness: float | None
    answer_relevancy: float | None
    trace: list[str] = field(default_factory=list)


@dataclass
class RagasBenchmarkReport:
    model: str
    embedder: str
    vector_backend: str
    total_answerable: int
    evaluated_samples: int
    answered_count: int
    handoff_count: int
    mean_faithfulness: float
    mean_answer_relevancy: float
    samples: list[RagasSampleScore] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "embedder": self.embedder,
            "vector_backend": self.vector_backend,
            "total_answerable": self.total_answerable,
            "evaluated_samples": self.evaluated_samples,
            "answered_count": self.answered_count,
            "handoff_count": self.handoff_count,
            "mean_faithfulness": self.mean_faithfulness,
            "mean_answer_relevancy": self.mean_answer_relevancy,
            "samples": [asdict(s) for s in self.samples],
        }

    def to_markdown(self) -> str:
        lines = [
            "# Sanad v2 — RAGAS Evaluation Report",
            "",
            f"- **LLM / Judge:** `{self.model}`",
            f"- **Embedder:** `{self.embedder}`",
            f"- **Vector DB:** `{self.vector_backend}`",
            f"- **Samples Evaluated:** `{self.evaluated_samples}` / `{self.total_answerable}` answerable questions (`answered={self.answered_count}`, `handoff={self.handoff_count}`)",
            "",
            "| RAGAS Metric | Score (0.0–1.0) |",
            "| --- | ---: |",
            f"| Faithfulness | {self.mean_faithfulness:.4f} |",
            f"| Answer Relevancy | {self.mean_answer_relevancy:.4f} |",
            "",
        ]
        return "\n".join(lines)


def _clean_metric(val: float | None) -> float | None:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return round(float(val), 4)


def run_ragas_benchmark(
    assistant: Assistant,
    questions: list[dict[str, Any]],
    *,
    ragas_llm: InstructorBaseRagasLLM,
    ragas_embeddings: BaseRagasEmbedding,
    vector_backend: str = "qdrant",
    judge_name: str = "ollama",
    limit: int | None = None,
    strictness: int = 1,
) -> tuple[RagasBenchmarkReport, EvaluationDataset]:
    """Run Sanad v2 on answerable golden questions and score with RAGAS Faithfulness & AnswerRelevancy."""
    answerable = [item for item in questions if item.get("expect") == "answer"]
    if limit is not None and limit > 0:
        answerable = answerable[:limit]

    faith_scorer = Faithfulness(llm=ragas_llm)
    rel_scorer = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=strictness)

    ragas_samples: list[SingleTurnSample] = []
    scored_rows: list[RagasSampleScore] = []
    answered_count = 0
    handoff_count = 0

    for item in answerable:
        q = item["question"]
        res = assistant.answer(q)
        lang = res.language or detect_language(q)
        hits = assistant.index.search(q, assistant.embedder, top_k=assistant.top_k, lang=lang)
        contexts = [h.chunk.text for h in hits]

        sample = SingleTurnSample(
            user_input=q,
            retrieved_contexts=contexts,
            response=res.answer,
        )
        ragas_samples.append(sample)

        if res.handoff is not None:
            handoff_count += 1
            scored_rows.append(
                RagasSampleScore(
                    question=q,
                    language=lang,
                    response=res.answer,
                    retrieved_contexts=contexts,
                    handed_off=True,
                    faithfulness=None,
                    answer_relevancy=None,
                    trace=list(res.trace),
                )
            )
            continue

        answered_count += 1
        faith_res = faith_scorer.score(
            user_input=q,
            response=res.answer,
            retrieved_contexts=contexts,
        )
        rel_res = rel_scorer.score(
            user_input=q,
            response=res.answer,
        )
        scored_rows.append(
            RagasSampleScore(
                question=q,
                language=lang,
                response=res.answer,
                retrieved_contexts=contexts,
                handed_off=False,
                faithfulness=_clean_metric(faith_res.value),
                answer_relevancy=_clean_metric(rel_res.value),
                trace=list(res.trace),
            )
        )

    faith_vals = [r.faithfulness for r in scored_rows if r.faithfulness is not None]
    rel_vals = [r.answer_relevancy for r in scored_rows if r.answer_relevancy is not None]
    mean_faith = round(sum(faith_vals) / len(faith_vals), 4) if faith_vals else 0.0
    mean_rel = round(sum(rel_vals) / len(rel_vals), 4) if rel_vals else 0.0

    report = RagasBenchmarkReport(
        model=f"{assistant.llm.name} (judge={judge_name})",
        embedder=assistant.embedder.name,
        vector_backend=vector_backend,
        total_answerable=len(answerable),
        evaluated_samples=len(scored_rows),
        answered_count=answered_count,
        handoff_count=handoff_count,
        mean_faithfulness=mean_faith,
        mean_answer_relevancy=mean_rel,
        samples=scored_rows,
    )
    return report, EvaluationDataset(samples=ragas_samples)


def save_ragas_report(report: RagasBenchmarkReport, json_path: Path, md_path: Path | None = None) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if md_path is not None:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(report.to_markdown(), encoding="utf-8")

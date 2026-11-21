import json
from typing import Type, TypeVar

from fastapi.testclient import TestClient
from pydantic import BaseModel
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections.answer_relevancy.metric import AnswerRelevanceOutput
from ragas.metrics.collections.faithfulness.util import (
    NLIStatementOutput,
    StatementFaithfulnessAnswer,
    StatementGeneratorOutput,
)

from mock_backend.app import create_app as create_backend
from sanad.assistant import Assistant
from sanad.backend import BackendClient
from sanad.eval import load_questions
from sanad.llm import ExtractiveLLM
from sanad.qdrant_store import QdrantVectorIndex
from sanad.ragas_eval import SanadRagasEmbedding, run_ragas_benchmark, save_ragas_report
from sanad.tools import BackendTools

from .conftest import ROOT

TModel = TypeVar("TModel", bound=BaseModel)


class DeterministicFakeRagasLLM(InstructorBaseRagasLLM):
    """Offline structured judge returning deterministic outputs for RAGAS Faithfulness & AnswerRelevancy."""

    def __init__(self):
        self.prompts: list[str] = []

    def _build(self, prompt: str, response_model: Type[TModel]) -> TModel:
        self.prompts.append(prompt)
        if response_model is StatementGeneratorOutput:
            return response_model(statements=["Shipping inside Cairo costs 35 EGP."])
        if response_model is NLIStatementOutput:
            return response_model(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="Shipping inside Cairo costs 35 EGP.",
                        reason="Directly supported by retrieved passage.",
                        verdict=1,
                    )
                ]
            )
        if response_model is AnswerRelevanceOutput:
            return response_model(question="مصاريف الشحن كام؟", noncommittal=0)
        raise AssertionError(f"Unexpected response_model: {response_model}")

    def generate(self, prompt: str, response_model: Type[TModel]) -> TModel:
        return self._build(prompt, response_model)

    async def agenerate(self, prompt: str, response_model: Type[TModel]) -> TModel:
        return self._build(prompt, response_model)


def test_ragas_benchmark_runs_end_to_end_offline(tmp_path, chunks, embedder, handoffs):
    q_idx = QdrantVectorIndex.build(chunks, embedder, in_memory=True)
    try:
        tools = BackendTools(BackendClient("http://testserver", client=TestClient(create_backend(api_key=""))))
        assistant = Assistant(
            q_idx,
            embedder,
            ExtractiveLLM(),
            handoffs,
            top_k=4,
            min_score=0.15,
            tools=tools,
            grounding_retries=1,
            strict_grounding=True,
        )
        questions = load_questions(ROOT / "data" / "eval" / "questions.jsonl")
        fake_judge = DeterministicFakeRagasLLM()
        ragas_emb = SanadRagasEmbedding(embedder)

        report, dataset = run_ragas_benchmark(
            assistant,
            questions,
            ragas_llm=fake_judge,
            ragas_embeddings=ragas_emb,
            vector_backend="qdrant",
            judge_name="fake-offline",
            limit=3,
            strictness=1,
        )

        assert report.evaluated_samples == 3
        assert len(dataset.samples) == 3
        assert report.answered_count == 3
        assert report.mean_faithfulness == 1.0
        assert report.mean_answer_relevancy > 0.0

        out_json = tmp_path / "ragas_report.json"
        out_md = tmp_path / "ragas_report.md"
        save_ragas_report(report, out_json, out_md)
        loaded = json.loads(out_json.read_text(encoding="utf-8"))
        assert loaded["vector_backend"] == "qdrant"
        assert loaded["mean_faithfulness"] == 1.0
        assert "RAGAS Evaluation Report" in out_md.read_text(encoding="utf-8")
    finally:
        q_idx.close()

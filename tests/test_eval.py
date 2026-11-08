from fastapi.testclient import TestClient

from mock_backend.app import create_app
from sanad.backend import BackendClient
from sanad.eval import evaluate, load_questions
from sanad.llm import ExtractiveLLM
from sanad.tools import BackendTools

from .conftest import ROOT


def test_eval_runs_on_golden_set(make_assistant):
    tools = BackendTools(BackendClient("http://testserver", client=TestClient(create_app(api_key=""))))
    assistant = make_assistant(ExtractiveLLM(), tools=tools, min_score=0.15)
    questions = load_questions(ROOT / "data" / "eval" / "questions.jsonl")
    summary = evaluate(assistant, questions).summary()
    assert summary["questions"] == len(questions) >= 30
    # Offline baseline (hash embedder + extractive LLM); guards against regressions.
    assert summary["retrieval_hit_at_k"] >= 0.95
    assert summary["handoff_recall"] == 1.0
    assert summary["tool_routing_accuracy"] == 1.0
    assert summary["answer_rate"] >= 0.85

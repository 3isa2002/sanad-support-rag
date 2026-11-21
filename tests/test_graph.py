from fastapi.testclient import TestClient

from mock_backend.app import create_app as create_backend
from sanad.assistant import Assistant
from sanad.backend import BackendClient
from sanad.llm import ScriptedLLM
from sanad.qdrant_store import QdrantVectorIndex
from sanad.tools import BackendTools


def test_langgraph_self_correction_fixes_missing_citation(chunks, embedder, handoffs):
    q_idx = QdrantVectorIndex.build(chunks, embedder, in_memory=True)
    try:
        llm = ScriptedLLM(
            "الشحن داخل القاهرة والجيزة بـ 35 جنيه.",  # missing [1] citation -> triggers retry
            "الشحن داخل القاهرة والجيزة بـ 35 جنيه [1].",  # corrected on attempt 2
        )
        assistant = Assistant(
            q_idx,
            embedder,
            llm,
            handoffs,
            top_k=4,
            min_score=0.15,
            grounding_retries=1,
            strict_grounding=True,
        )
        res = assistant.answer("مصاريف الشحن كام؟")
        assert res.handoff is None
        assert res.sources and res.sources[0].doc_id == "shipping-ar"
        assert len(llm.calls) == 2
        assert "IMPORTANT CORRECTION" in llm.calls[1][-1]["content"]
        assert res.trace == [
            "router",
            "retriever",
            "generator",
            "grounding_checker",
            "generator",
            "grounding_checker",
            "responder",
        ]
    finally:
        q_idx.close()


def test_langgraph_catches_hallucinated_number_and_hands_off(chunks, embedder, handoffs):
    q_idx = QdrantVectorIndex.build(chunks, embedder, in_memory=True)
    try:
        llm = ScriptedLLM(
            "الشحن داخل القاهرة بـ 9999 جنيه [1].",
            "الشحن بـ 8888 جنيه [1].",
        )
        assistant = Assistant(
            q_idx,
            embedder,
            llm,
            handoffs,
            top_k=4,
            min_score=0.15,
            grounding_retries=1,
            strict_grounding=True,
        )
        res = assistant.answer("مصاريف الشحن كام؟")
        assert res.handoff is not None
        assert res.handoff.reason == "grounding_failed"
        assert any("ungrounded_numbers" in item for item in res.grounding_issues)
        assert res.trace[-1] == "handoff"
        ticket = handoffs.get(res.handoff.ticket_id)
        assert ticket.context["agent_trace"][-1] == "handoff"
    finally:
        q_idx.close()


def test_langgraph_catches_phantom_citation(chunks, embedder, handoffs):
    q_idx = QdrantVectorIndex.build(chunks, embedder, in_memory=True)
    try:
        llm = ScriptedLLM("Shipping is 35 EGP [1][9].")
        assistant = Assistant(
            q_idx,
            embedder,
            llm,
            handoffs,
            top_k=4,
            min_score=0.15,
            grounding_retries=0,
            strict_grounding=True,
        )
        res = assistant.answer("how much is shipping inside Cairo?")
        assert res.handoff is not None
        assert res.handoff.reason == "grounding_failed"
        assert "phantom_citation:[9]" in res.grounding_issues
    finally:
        q_idx.close()


def test_langgraph_multiturn_checkpointer_remembers_order_ref(chunks, embedder, handoffs):
    q_idx = QdrantVectorIndex.build(chunks, embedder, in_memory=True)
    try:
        backend = BackendClient("http://testserver", client=TestClient(create_backend(api_key="")))
        tools = BackendTools(backend)
        assistant = Assistant(q_idx, embedder, ScriptedLLM(), handoffs, min_score=0.15, tools=tools)

        t1 = assistant.answer("فين الاوردر ORD-10023؟", conversation_id="conv-multi-1")
        assert t1.tool_calls and t1.tool_calls[0].arguments["id"] == "ORD-10023"
        assert t1.trace == ["router", "tool"]

        # Turn 2 omits the order ID, relying on LangGraph's checkpointed state for conv-multi-1.
        t2 = assistant.answer("فين الاوردر بتاعي؟", conversation_id="conv-multi-1")
        assert t2.handoff is None
        assert t2.tool_calls and t2.tool_calls[0].arguments["id"] == "ORD-10023"
        assert "SQ7781203EG" in t2.answer
        assert t2.trace == ["router", "tool"]
    finally:
        q_idx.close()


def test_langgraph_mermaid_export(index, embedder, handoffs):
    assistant = Assistant(index, embedder, ScriptedLLM(), handoffs)
    diagram = assistant.graph_mermaid()
    for node in ("router_agent", "tool_agent", "retriever_agent", "generator_agent", "grounding_checker"):
        assert node in diagram

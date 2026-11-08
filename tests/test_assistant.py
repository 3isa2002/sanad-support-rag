import json

import httpx
import pytest

from sanad.assistant import is_handoff_reply, parse_citations, wants_human
from sanad.llm import ExtractiveLLM, LLMError, OllamaChat, ScriptedLLM


# ---------------------------------------------------------------- pure helpers

@pytest.mark.parametrize(
    "text",
    ["عايز أكلم موظف", "ممكن اتكلم مع حد من خدمة العملاء", "حولني لموظف", "موظف",
     "I want to talk to a human", "can I speak with an agent?", "agent", "live agent please"],
)
def test_wants_human_positive(text):
    assert wants_human(text)


@pytest.mark.parametrize(
    "text",
    ["ازاي ارجع منتج", "المندوب مجاش", "عايز مندوب ييجي ياخد المرتجع", "How do refunds work?",
     "is the agent fee included?"],
)
def test_wants_human_negative(text):
    assert not wants_human(text)


def test_parse_citations():
    assert parse_citations("A [1]. B [3][1]. C [2, 4]", 3) == [1, 3, 2]
    assert parse_citations("مثال [١]", 2) == [1]
    assert parse_citations("no refs [9]", 3) == []


def test_is_handoff_reply():
    assert is_handoff_reply("HANDOFF")
    assert is_handoff_reply("handoff.")
    assert is_handoff_reply("Sorry. HANDOFF")
    assert not is_handoff_reply("Refunds take 5 days [1]")


# ---------------------------------------------------------------- flows

def test_grounded_answer_with_sources(make_assistant):
    llm = ScriptedLLM("استرداد الكارت بياخد من 5 لـ 10 أيام عمل [1].")
    res = make_assistant(llm).answer("الفلوس بترجع امتى لو دفعت بالكارت؟")
    assert res.handoff is None
    assert res.language == "ar"
    assert res.sources and res.sources[0].doc_id == "returns-policy-ar"
    # Prompt contains numbered context and the language instruction.
    user_prompt = llm.calls[0][-1]["content"]
    assert "[1]" in user_prompt and "Reply in Arabic" in user_prompt


def test_only_cited_sources_returned(make_assistant):
    res = make_assistant(ScriptedLLM("Card refunds take 5–10 business days [2].")).answer(
        "how long does a card refund take?"
    )
    assert len(res.sources) == 1


def test_low_confidence_hands_off_without_calling_llm(make_assistant, handoffs):
    llm = ScriptedLLM()
    res = make_assistant(llm).answer("what is the capital of France")
    assert res.handoff and res.handoff.reason == "low_confidence"
    assert llm.calls == []
    ticket = handoffs.get(res.handoff.ticket_id)
    assert ticket.question == "what is the capital of France"
    assert "best_score" in ticket.context
    assert res.handoff.ticket_id in res.answer


def test_model_uncertain_hands_off(make_assistant):
    res = make_assistant(ScriptedLLM("HANDOFF")).answer("مصاريف الشحن كام للاسكندرية؟")
    assert res.handoff.reason == "model_uncertain"
    assert res.language == "ar" and "رقم التذكرة" in res.answer
    assert res.sources == []


def test_uncited_answer_hands_off(make_assistant, handoffs):
    res = make_assistant(ScriptedLLM("Refunds are quick, don't worry!")).answer("how long does a card refund take?")
    assert res.handoff.reason == "no_citation"
    assert handoffs.get(res.handoff.ticket_id).context["draft_answer"].startswith("Refunds")


def test_llm_error_hands_off(make_assistant):
    res = make_assistant(ScriptedLLM(LLMError("down"))).answer("how long does a card refund take?")
    assert res.handoff.reason == "llm_error"


def test_human_request_hands_off(make_assistant):
    llm = ScriptedLLM()
    res = make_assistant(llm).answer("I want to talk to a human")
    assert res.handoff.reason == "human_requested" and res.language == "en"
    assert llm.calls == []


def test_empty_question(make_assistant):
    res = make_assistant(ScriptedLLM()).answer("   ")
    assert res.handoff is None and res.answer


def test_extractive_llm_cites(make_assistant):
    res = make_assistant(ExtractiveLLM()).answer("مصاريف الشحن كام")
    assert res.handoff is None
    assert "35" in res.answer and res.sources[0].section == "مصاريف الشحن"


# ---------------------------------------------------------------- handoff store

def test_handoff_store_roundtrip(handoffs):
    t1 = handoffs.create("q1", "ar", "low_confidence", "conv-1", {"x": 1})
    t2 = handoffs.create("q2", "en", "human_requested")
    assert t1.id == "HT-000001" and t2.id == "HT-000002"
    assert [t.id for t in handoffs.list()] == ["HT-000002", "HT-000001"]
    assert handoffs.get("HT-000001").context == {"x": 1}
    assert handoffs.get("bogus") is None
    with pytest.raises(ValueError):
        handoffs.create("q", "en", "made_up_reason")


# ---------------------------------------------------------------- ollama client

def test_ollama_chat_request_and_parse():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": " hi [1] "}})

    client = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))
    out = OllamaChat("http://ollama", "qwen2.5:7b", client=client).chat([{"role": "user", "content": "x"}])
    assert out == "hi [1]"
    assert seen["model"] == "qwen2.5:7b" and seen["stream"] is False


def test_ollama_chat_error_raises_llm_error():
    client = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    with pytest.raises(LLMError):
        OllamaChat("http://ollama", "m", client=client).chat([{"role": "user", "content": "x"}])

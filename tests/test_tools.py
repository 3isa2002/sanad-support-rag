import httpx
import pytest
from fastapi.testclient import TestClient

from mock_backend.app import create_app
from sanad.backend import BackendClient, BackendError, NotFound
from sanad.llm import ScriptedLLM
from sanad.tools import BackendTools, Ref, extract_refs, status_intent_without_id

API_KEY = "test-key"


@pytest.fixture
def backend():
    return BackendClient("http://testserver", api_key=API_KEY, client=TestClient(create_app(api_key=API_KEY)))


@pytest.fixture
def tools(backend):
    return BackendTools(backend)


# ---------------------------------------------------------------- extraction

@pytest.mark.parametrize(
    "text, expected",
    [
        ("فين الاوردر ORD-10023؟", [Ref("order", "ORD-10023")]),
        ("Where is my order ord10024", [Ref("order", "ORD-10024")]),
        ("الطلب رقم ١٠٠٢٥ اتأخر", [Ref("order", "ORD-10025")]),
        ("my order #10026 status", [Ref("order", "ORD-10026")]),
        ("إعلاني LST-5003 اترفض ليه", [Ref("listing", "LST-5003")]),
        ("الإعلان رقم 5002", [Ref("listing", "LST-5002")]),
        ("ORD-10023 and LST-5001", [Ref("order", "ORD-10023"), Ref("listing", "LST-5001")]),
        ("ازاي ارجع منتج", []),
        ("Shipping is free for orders above 500 EGP", []),
        ("الطلبات فوق 1500 جنيه بتتقسط؟", []),
    ],
)
def test_extract_refs(text, expected):
    assert extract_refs(text) == expected


def test_status_intent_without_id():
    assert status_intent_without_id("فين الاوردر بتاعي؟") == "order"
    assert status_intent_without_id("where is my order?") == "order"
    assert status_intent_without_id("اعلاني اترفض") == "listing"
    assert status_intent_without_id("مصاريف الشحن كام") is None


# ---------------------------------------------------------------- client

def test_client_get_and_errors(backend):
    assert backend.get_order("ORD-10023")["status"] == "shipped"
    with pytest.raises(NotFound):
        backend.get_order("ORD-99999")


def test_client_requires_api_key():
    client = BackendClient("http://testserver", client=TestClient(create_app(api_key="secret")))
    with pytest.raises(BackendError, match="401"):
        client.get_order("ORD-10023")


def test_client_unreachable():
    def boom(request):
        raise httpx.ConnectError("refused")

    client = BackendClient("http://x", client=httpx.Client(base_url="http://x", transport=httpx.MockTransport(boom)))
    with pytest.raises(BackendError, match="unreachable"):
        client.get_listing("LST-5001")


# ---------------------------------------------------------------- assistant + tools

def test_order_status_arabic(make_assistant, tools):
    llm = ScriptedLLM()
    res = make_assistant(llm, tools=tools).answer("فين الاوردر ORD-10023؟")
    assert res.language == "ar" and res.handoff is None
    assert "تم الشحن" in res.answer and "SQ7781203EG" in res.answer and "2026-11-09" in res.answer
    assert res.tool_calls[0].name == "get_order_status" and res.tool_calls[0].ok
    assert llm.calls == []  # facts come from the backend, not the model


def test_order_status_english_delivered(make_assistant, tools):
    res = make_assistant(ScriptedLLM(), tools=tools).answer("Where is my order ORD-10024?")
    assert "Delivered" in res.answer and "Estimated delivery" not in res.answer


def test_listing_rejected_reason(make_assistant, tools):
    res = make_assistant(ScriptedLLM(), tools=tools).answer("إعلاني LST-5003 اترفض ليه؟")
    assert "مرفوض" in res.answer and "بيانات تواصل" in res.answer


def test_not_found_asks_to_recheck(make_assistant, tools, handoffs):
    res = make_assistant(ScriptedLLM(), tools=tools).answer("order ORD-77777 status")
    assert res.handoff is None and "ORD-77777" in res.answer
    assert res.tool_calls[0].error == "not_found"
    assert handoffs.list() == []


def test_backend_down_hands_off(make_assistant):
    def boom(request):
        raise httpx.ConnectError("refused")

    down = BackendTools(BackendClient("http://x", client=httpx.Client(base_url="http://x", transport=httpx.MockTransport(boom))))
    res = make_assistant(ScriptedLLM(), tools=down).answer("فين الطلب ORD-10023")
    assert res.handoff.reason == "backend_error"
    assert "الطلب" in res.answer and res.handoff.ticket_id in res.answer
    assert res.tool_calls and not res.tool_calls[0].ok


def test_status_question_without_id_asks_for_it(make_assistant, tools):
    res = make_assistant(ScriptedLLM(), tools=tools).answer("فين الاوردر بتاعي؟")
    assert "رقم الطلب" in res.answer and res.handoff is None


def test_policy_question_falls_through_to_rag(make_assistant, tools):
    res = make_assistant(ScriptedLLM("Shipping costs 35 EGP inside Cairo [1]."), tools=tools).answer(
        "how much is shipping inside Cairo?"
    )
    assert res.tool_calls == [] and res.sources

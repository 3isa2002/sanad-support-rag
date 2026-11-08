import pytest
from fastapi.testclient import TestClient

from mock_backend.app import create_app as create_backend
from sanad.api import create_app
from sanad.assistant import Assistant
from sanad.backend import BackendClient
from sanad.config import Settings
from sanad.llm import ExtractiveLLM
from sanad.tools import BackendTools

from .conftest import KB_DIR


@pytest.fixture
def settings(tmp_path, monkeypatch):
    for var in ("SANAD_ADMIN_TOKEN", "SANAD_MIN_SCORE"):
        monkeypatch.delenv(var, raising=False)
    s = Settings()
    s.kb_dir, s.index_dir, s.db_path = KB_DIR, tmp_path / "index", tmp_path / "sanad.db"
    s.embedder, s.llm, s.admin_token = "hash", "fake", "admin-secret"
    return s


@pytest.fixture
def client(settings, index, embedder, handoffs):
    tools = BackendTools(BackendClient("http://testserver", client=TestClient(create_backend(api_key=""))))
    assistant = Assistant(index, embedder, ExtractiveLLM(), handoffs, min_score=0.25, tools=tools)
    return TestClient(create_app(settings, assistant))


ADMIN = {"X-Admin-Token": "admin-secret"}


def test_chat_rag_answer(client):
    r = client.post("/chat", json={"message": "مصاريف الشحن كام؟"})
    assert r.status_code == 200
    body = r.json()
    assert body["language"] == "ar" and body["handoff"] is None
    assert body["sources"][0]["doc_id"] == "shipping-ar"


def test_chat_order_tool(client):
    body = client.post("/chat", json={"message": "where is my order ORD-10027"}).json()
    assert "Out for delivery" in body["answer"]
    assert body["tool_calls"][0]["result"]["tracking_number"] == "SQ7782456EG"


def test_chat_handoff_then_list_tickets(client):
    body = client.post("/chat", json={"message": "talk to a human please", "conversation_id": "c-1"}).json()
    ticket_id = body["handoff"]["ticket_id"]
    assert client.get("/handoffs").status_code == 401
    tickets = client.get("/handoffs", headers=ADMIN).json()
    assert tickets[0]["id"] == ticket_id and tickets[0]["conversation_id"] == "c-1"
    assert client.get(f"/handoffs/{ticket_id}", headers=ADMIN).json()["reason"] == "human_requested"
    assert client.get("/handoffs/HT-999999", headers=ADMIN).status_code == 404


def test_chat_validation(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert client.post("/chat", json={"message": "x" * 2001}).status_code == 422


def test_reindex_requires_admin(client, settings):
    assert client.post("/admin/reindex").status_code == 401
    r = client.post("/admin/reindex", headers=ADMIN)
    assert r.status_code == 200 and r.json()["documents"] == 10
    assert (settings.index_dir / "meta.json").is_file()


def test_health_and_index_page(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["embedder"].startswith("hash") and h["backend_reachable"] is True
    page = client.get("/")
    assert page.status_code == 200 and "سَنَد" in page.text


def test_create_app_builds_missing_index(settings):
    app = create_app(settings)  # wires real factory; index dir is empty → built on startup
    assert (settings.index_dir / "vectors.npy").is_file()
    body = TestClient(app).post("/chat", json={"message": "how much is shipping inside Cairo?"}).json()
    assert body["sources"]

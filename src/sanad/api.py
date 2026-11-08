"""FastAPI application: chat endpoint, handoff tickets, health and reindex."""

from __future__ import annotations

import logging
import secrets
import threading
from importlib.resources import files

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .assistant import Assistant, AssistantResponse
from .config import Settings
from .embeddings import make_embedder
from .index import HybridIndex
from .kb import build_chunks

log = logging.getLogger("sanad")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: str | None = Field(default=None, max_length=100)


def load_or_build_index(settings: Settings) -> HybridIndex:
    embedder = make_embedder(settings)
    try:
        return HybridIndex.load(settings.index_dir, expected_embedder=embedder.name)
    except (FileNotFoundError, ValueError) as exc:
        log.warning("%s — building the index from %s", exc, settings.kb_dir)
        index = HybridIndex.build(build_chunks(settings.kb_dir), embedder)
        index.save(settings.index_dir)
        return index


def create_app(settings: Settings | None = None, assistant: Assistant | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    if assistant is None:
        from .factory import build_assistant

        assistant = build_assistant(settings, index=load_or_build_index(settings))

    app = FastAPI(title="Sanad support assistant", version="0.1.0")
    app.state.assistant = assistant
    reindex_lock = threading.Lock()

    def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
        token = settings.admin_token
        if token and not (x_admin_token and secrets.compare_digest(x_admin_token, token)):
            raise HTTPException(status_code=401, detail="admin token required (X-Admin-Token header)")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index_page() -> str:
        return (files("sanad") / "static" / "index.html").read_text(encoding="utf-8")

    @app.post("/chat", response_model=AssistantResponse)
    def chat(req: ChatRequest) -> AssistantResponse:
        return app.state.assistant.answer(req.message, conversation_id=req.conversation_id)

    @app.get("/handoffs", dependencies=[Depends(require_admin)])
    def list_handoffs(status: str | None = None, limit: int = Query(50, ge=1, le=500)) -> list[dict]:
        return [t.to_dict() for t in app.state.assistant.handoffs.list(status=status, limit=limit)]

    @app.get("/handoffs/{ticket_id}", dependencies=[Depends(require_admin)])
    def get_handoff(ticket_id: str) -> dict:
        ticket = app.state.assistant.handoffs.get(ticket_id)
        if not ticket:
            raise HTTPException(status_code=404, detail="ticket not found")
        return ticket.to_dict()

    @app.post("/admin/reindex", dependencies=[Depends(require_admin)])
    def reindex() -> dict:
        with reindex_lock:
            a = app.state.assistant
            index = HybridIndex.build(build_chunks(settings.kb_dir), a.embedder)
            index.save(settings.index_dir)
            a.index = index
        return {"chunks": len(index.chunks), "documents": len({c.doc_id for c in index.chunks})}

    @app.get("/health")
    def health() -> dict:
        a = app.state.assistant
        llm_ok = a.llm.ping() if hasattr(a.llm, "ping") else True
        backend_ok = a.tools.client.ping() if a.tools is not None and hasattr(a.tools, "client") else None
        return {
            "status": "ok" if llm_ok and backend_ok is not False else "degraded",
            "llm": {"name": a.llm.name, "reachable": llm_ok},
            "embedder": a.embedder.name,
            "index_chunks": len(a.index.chunks),
            "min_score": a.min_score,
            "backend_reachable": backend_ok,
        }

    return app

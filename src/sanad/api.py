"""FastAPI application: chat endpoint, handoff tickets, health and reindex."""

from __future__ import annotations

import logging
import secrets
import threading
from importlib.resources import files

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .assistant import Assistant, AssistantResponse
from .config import Settings
from .embeddings import make_embedder
from .kb import build_chunks
from .qdrant_store import (
    VectorIndex,
    build_vector_index,
    load_or_build_vector_index,
    save_vector_index,
)
from .tracing import AgentTracer

log = logging.getLogger("sanad")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: str | None = Field(default=None, max_length=100)


def load_or_build_index(settings: Settings) -> VectorIndex:
    embedder = make_embedder(settings)
    return load_or_build_vector_index(settings, embedder)


def create_app(settings: Settings | None = None, assistant: Assistant | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    if assistant is None:
        from .factory import build_assistant

        assistant = build_assistant(settings, index=load_or_build_index(settings))

    app = FastAPI(title="Sanad support assistant", version="0.3.0")
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()] or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def allow_private_network_preflight(request: Request, call_next):
        response = await call_next(request)
        if request.headers.get("access-control-request-private-network") == "true":
            response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

    app.state.assistant = assistant
    default_tracer = assistant.tracer or AgentTracer.from_settings(settings)
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
            if hasattr(a.index, "close"):
                a.index.close()
            index = build_vector_index(settings, build_chunks(settings.kb_dir), a.embedder)
            save_vector_index(index, settings.index_dir)
            a.index = index
        return {
            "chunks": len(index.chunks),
            "documents": len({c.doc_id for c in index.chunks}),
            "vector_backend": settings.vector_backend,
        }

    @app.get("/traces", dependencies=[Depends(require_admin)])
    def list_traces(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
        tracer = app.state.assistant.tracer or default_tracer
        return tracer.recent(limit=limit)

    @app.get("/metrics")
    def metrics() -> dict:
        tracer = app.state.assistant.tracer or default_tracer
        return tracer.summary_metrics()

    @app.get("/health")
    def health() -> dict:
        a = app.state.assistant
        tracer = a.tracer or default_tracer
        llm_ok = a.llm.ping() if hasattr(a.llm, "ping") else True
        backend_ok = a.tools.client.ping() if a.tools is not None and hasattr(a.tools, "client") else None
        qdrant_ok = a.index.ping() if hasattr(a.index, "ping") else None
        return {
            "status": "ok" if llm_ok and backend_ok is not False and qdrant_ok is not False else "degraded",
            "llm": {"name": a.llm.name, "reachable": llm_ok},
            "embedder": a.embedder.name,
            "vector_backend": settings.vector_backend,
            "qdrant_reachable": qdrant_ok,
            "tracing": tracer.backend,
            "index_chunks": len(a.index.chunks),
            "min_score": a.min_score,
            "backend_reachable": backend_ok,
        }

    return app

"""SQLite-backed store for human-handoff tickets."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

REASONS = {
    "human_requested",
    "low_confidence",
    "model_uncertain",
    "no_citation",
    "llm_error",
    "backend_error",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS handoffs (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    conversation_id TEXT,
    language TEXT NOT NULL,
    question TEXT NOT NULL,
    reason TEXT NOT NULL,
    context TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'open'
)
"""


@dataclass
class HandoffTicket:
    id: str
    created_at: str
    conversation_id: str | None
    language: str
    question: str
    reason: str
    context: dict
    status: str = "open"

    def to_dict(self) -> dict:
        return asdict(self)


def _ticket_id(seq: int) -> str:
    return f"HT-{seq:06d}"


class HandoffStore:
    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute(_SCHEMA)

    def create(
        self,
        question: str,
        language: str,
        reason: str,
        conversation_id: str | None = None,
        context: dict | None = None,
    ) -> HandoffTicket:
        if reason not in REASONS:
            raise ValueError(f"Unknown handoff reason: {reason}")
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        ctx = context or {}
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO handoffs (created_at, conversation_id, language, question, reason, context)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (created_at, conversation_id, language, question, reason, json.dumps(ctx, ensure_ascii=False)),
            )
            seq = cur.lastrowid
        return HandoffTicket(_ticket_id(seq), created_at, conversation_id, language, question, reason, ctx)

    def list(self, status: str | None = None, limit: int = 50) -> list[HandoffTicket]:
        query = "SELECT * FROM handoffs"
        params: tuple = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY seq DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(query, (*params, limit)).fetchall()
        return [self._row(r) for r in rows]

    def get(self, ticket_id: str) -> HandoffTicket | None:
        try:
            seq = int(ticket_id.removeprefix("HT-"))
        except ValueError:
            return None
        with self._lock:
            row = self._conn.execute("SELECT * FROM handoffs WHERE seq = ?", (seq,)).fetchone()
        return self._row(row) if row else None

    @staticmethod
    def _row(row: sqlite3.Row) -> HandoffTicket:
        return HandoffTicket(
            id=_ticket_id(row["seq"]),
            created_at=row["created_at"],
            conversation_id=row["conversation_id"],
            language=row["language"],
            question=row["question"],
            reason=row["reason"],
            context=json.loads(row["context"]),
            status=row["status"],
        )

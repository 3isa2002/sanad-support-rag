"""Settings loaded from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Tiny .env loader: KEY=VALUE lines, existing env vars win."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# Cosine-similarity thresholds below which retrieval is considered too weak to answer.
# The hash embedder produces lower similarities than a real semantic model.
DEFAULT_MIN_SCORE = {"hash": 0.15, "ollama": 0.45}


@dataclass
class Settings:
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    chat_model: str = field(default_factory=lambda: _env("SANAD_CHAT_MODEL", "qwen2.5:7b"))
    embed_model: str = field(default_factory=lambda: _env("SANAD_EMBED_MODEL", "bge-m3"))
    embedder: str = field(default_factory=lambda: _env("SANAD_EMBEDDER", "ollama"))
    llm: str = field(default_factory=lambda: _env("SANAD_LLM", "ollama"))
    kb_dir: Path = field(default_factory=lambda: Path(_env("SANAD_KB_DIR", "data/kb")))
    index_dir: Path = field(default_factory=lambda: Path(_env("SANAD_INDEX_DIR", "data/index")))
    db_path: Path = field(default_factory=lambda: Path(_env("SANAD_DB_PATH", "data/sanad.db")))
    top_k: int = field(default_factory=lambda: int(_env("SANAD_TOP_K", "4")))
    # Minimum dense similarity of the best chunk; None → per-embedder default.
    min_score: float | None = field(
        default_factory=lambda: float(os.environ["SANAD_MIN_SCORE"]) if os.environ.get("SANAD_MIN_SCORE") else None
    )
    backend_url: str = field(default_factory=lambda: _env("SANAD_BACKEND_URL", "http://localhost:8001"))
    backend_api_key: str = field(default_factory=lambda: _env("SANAD_BACKEND_API_KEY", ""))
    backend_timeout: float = field(default_factory=lambda: float(_env("SANAD_BACKEND_TIMEOUT", "5")))
    admin_token: str = field(default_factory=lambda: _env("SANAD_ADMIN_TOKEN", ""))
    host: str = field(default_factory=lambda: _env("SANAD_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(_env("SANAD_PORT", "8000")))
    llm_timeout: float = field(default_factory=lambda: float(_env("SANAD_LLM_TIMEOUT", "120")))

    def effective_min_score(self, embedder_name: str) -> float:
        if self.min_score is not None:
            return self.min_score
        return DEFAULT_MIN_SCORE["hash" if embedder_name.startswith("hash") else "ollama"]

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        return cls()

"""Chat LLM providers: Ollama (default), an offline extractive fallback, and a scripted fake for tests."""

from __future__ import annotations

import re
from typing import Protocol

import httpx


class LLMError(RuntimeError):
    pass


class ChatLLM(Protocol):
    name: str

    def chat(self, messages: list[dict[str, str]]) -> str: ...


class OllamaChat:
    """Calls Ollama's ``POST /api/chat`` with streaming disabled."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 120.0,
        temperature: float = 0.1,
        client: httpx.Client | None = None,
    ):
        self.model = model
        self.name = f"ollama:{model}"
        self.temperature = temperature
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout)

    def chat(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        try:
            resp = self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise LLMError(
                f"Ollama chat request failed ({exc}). Is Ollama running and is '{self.model}' pulled?"
            ) from exc

    def ping(self) -> bool:
        try:
            return self._client.get("/api/tags", timeout=3).status_code == 200
        except httpx.HTTPError:
            return False


class ExtractiveLLM:
    """Offline stand-in for demos without Ollama: returns the most relevant context
    passage verbatim with a citation. Never invents text; if there is no context it
    asks for a handoff."""

    name = "extractive"
    _CONTEXT = re.compile(r"^\[(\d+)\][^\n]*\n(.*?)(?=^\[\d+\]|\Z)", re.DOTALL | re.MULTILINE)

    def chat(self, messages: list[dict[str, str]]) -> str:
        user = messages[-1]["content"]
        if "<tool_result>" in user:
            data = user.split("<tool_result>", 1)[1].split("</tool_result>", 1)[0].strip()
            return data
        passages = self._CONTEXT.findall(user.split("<context>", 1)[-1].split("</context>", 1)[0])
        if not passages:
            return "HANDOFF"
        num, text = passages[0]
        return f"{text.strip()} [{num}]"


class ScriptedLLM:
    """Test double: returns queued responses (or raises queued exceptions) and records prompts."""

    name = "scripted"

    def __init__(self, *responses: str | Exception):
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("ScriptedLLM ran out of responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_llm(settings) -> ChatLLM:
    if settings.llm == "ollama":
        return OllamaChat(settings.ollama_base_url, settings.chat_model, settings.llm_timeout)
    if settings.llm in ("fake", "extractive"):
        return ExtractiveLLM()
    raise ValueError(f"Unknown SANAD_LLM: {settings.llm!r} (use 'ollama' or 'fake')")

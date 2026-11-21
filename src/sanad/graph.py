"""Stateful multi-agent LangGraph workflow for Sanad v2."""

from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .llm import LLMError
from .messages import msg
from .models import (
    AssistantResponse,
    Handoff,
    extract_all_citation_numbers,
    is_handoff_reply,
    parse_citations,
    source_from_result,
    wants_human,
)
from .prompts import grounded_messages, grounding_judge_messages
from .textnorm import detect_language, to_latin_digits
from .tools import extract_refs, status_intent_without_id

if TYPE_CHECKING:
    from .assistant import Assistant

_BRACKET_CITES = re.compile(r"\[[\d\s,،]+\]")
_LIST_PREFIX = re.compile(r"(?m)^\s*\d+[\.\)]\s+")
_COMMA_NUM = re.compile(r"(?<=\d),(?=\d)")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")

MAX_HISTORY_ITEMS = 6


def _trace_reducer(prev: list[str] | None, new: list[str] | None) -> list[str]:
    items = list(new or [])
    if "__RESET__" in items:
        return [x for x in items if x != "__RESET__"]
    return list(prev or []) + items


def _history_reducer(prev: list[dict[str, str]] | None, new: list[dict[str, str]] | None) -> list[dict[str, str]]:
    merged = list(prev or []) + list(new or [])
    return merged[-MAX_HISTORY_ITEMS:]


class SanadGraphState(TypedDict, total=False):
    question: str
    effective_query: str
    conversation_id: str | None
    language: str
    route: Literal["empty", "human", "tool", "rag"]
    history: Annotated[list[dict[str, str]], _history_reducer]
    last_ref: str | None
    retrieved: list[Any]
    best_score: float
    draft_answer: str
    cited_indices: list[int]
    grounding_ok: bool
    grounding_issues: list[str]
    attempts: int
    max_attempts: int
    handoff_reason: str | None
    handoff_key: str
    handoff_context: dict[str, Any]
    agent_trace: Annotated[list[str], _trace_reducer]
    final_response: AssistantResponse | None


def _extract_numbers(text: str) -> set[str]:
    cleaned = _COMMA_NUM.sub("", to_latin_digits(text))
    return set(_NUMBER.findall(cleaned))


def check_grounding(
    question: str,
    draft: str,
    retrieved: list[Any],
    lang: str,
    *,
    check_numbers: bool = True,
    llm: Any = None,
    use_llm_judge: bool = False,
) -> tuple[list[int], list[str]]:
    """Verify inline citations, numeric grounding, language alignment, and optional LLM critique."""
    issues: list[str] = []
    raw_nums = extract_all_citation_numbers(draft)
    phantom = [n for n in raw_nums if n < 1 or n > len(retrieved)]
    if phantom:
        issues.append(f"phantom_citation:{phantom}")

    cited = parse_citations(draft, len(retrieved))
    if not cited:
        issues.append("missing_citation")
        return cited, issues

    # Verify numbers in the draft appear in the cited passage texts.
    if check_numbers:
        draft_body = _LIST_PREFIX.sub("", _BRACKET_CITES.sub("", to_latin_digits(draft)))
        draft_nums = _extract_numbers(draft_body)
        cited_text = "\n".join(retrieved[i - 1].chunk.text for i in cited)
        cited_nums = _extract_numbers(cited_text)
        ungrounded_nums = sorted(n for n in draft_nums if n not in cited_nums)
        if ungrounded_nums:
            issues.append(f"ungrounded_numbers:{ungrounded_nums}")

    if len(draft.strip()) >= 8:
        detected = detect_language(draft)
        if detected != lang:
            issues.append(f"language_mismatch:expected_{lang}_got_{detected}")

    if not issues and use_llm_judge and llm is not None:
        try:
            verdict = llm.chat(grounding_judge_messages(question, draft, retrieved)).strip().upper()
            if verdict.startswith("UNSUPPORTED"):
                issues.append("llm_judge_unsupported")
        except LLMError:
            pass

    return cited, issues


def build_sanad_graph(assistant: "Assistant") -> Any:
    """Compile Sanad's stateful multi-agent LangGraph workflow with InMemorySaver checkpointer."""

    def router_agent(state: SanadGraphState) -> dict[str, Any]:
        raw_q = (state.get("question") or "").strip()
        lang = detect_language(raw_q)
        if not raw_q:
            return {
                "question": raw_q,
                "effective_query": raw_q,
                "language": lang,
                "route": "empty",
                "agent_trace": ["router"],
            }

        if wants_human(raw_q):
            return {
                "question": raw_q,
                "effective_query": raw_q,
                "language": lang,
                "route": "human",
                "handoff_reason": "human_requested",
                "handoff_key": "human_requested",
                "handoff_context": {},
                "agent_trace": ["router"],
            }

        eff_q = raw_q
        last_ref = state.get("last_ref")
        refs = extract_refs(eff_q)
        if not refs and last_ref:
            intent_kind = status_intent_without_id(eff_q)
            ref_matches_intent = (
                (intent_kind == "order" and last_ref.startswith("ORD-"))
                or (intent_kind == "listing" and last_ref.startswith("LST-"))
            )
            if ref_matches_intent:
                eff_q = f"{raw_q} ({last_ref})"
                refs = extract_refs(eff_q)

        if assistant.tools is not None and (refs or status_intent_without_id(eff_q)):
            return {
                "question": raw_q,
                "effective_query": eff_q,
                "language": lang,
                "route": "tool",
                "agent_trace": ["router"],
            }

        # Short multi-turn follow-up enrichment using checkpointed thread history.
        history = state.get("history") or []
        if history and len(raw_q.split()) <= 5:
            prev_user = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "")
            if prev_user:
                eff_q = f"{prev_user} — {raw_q}"

        return {
            "question": raw_q,
            "effective_query": eff_q,
            "language": lang,
            "route": "rag",
            "agent_trace": ["router"],
        }

    def tool_agent(state: SanadGraphState) -> dict[str, Any]:
        eff_q = state["effective_query"]
        lang = state["language"]
        conv_id = state.get("conversation_id")
        refs = extract_refs(eff_q)
        next_ref = refs[-1].id if refs else state.get("last_ref")

        resp = assistant.tools.handle(eff_q, lang, assistant, conv_id)
        if resp is None:
            return {"agent_trace": ["tool"], "last_ref": next_ref}

        trace = list(state.get("agent_trace") or []) + ["tool"]
        resp.trace = trace
        return {
            "final_response": resp,
            "last_ref": next_ref,
            "agent_trace": ["tool"],
            "history": [
                {"role": "user", "content": state["question"]},
                {"role": "assistant", "content": resp.answer},
            ],
        }

    def retriever_agent(state: SanadGraphState) -> dict[str, Any]:
        eff_q = state["effective_query"]
        lang = state["language"]
        results = assistant.index.search(eff_q, assistant.embedder, top_k=assistant.top_k, lang=lang)
        best = max((r.score for r in results), default=0.0)
        retrieved_meta = [source_from_result(r).model_dump() for r in results]

        if best < assistant.min_score:
            return {
                "retrieved": results,
                "best_score": best,
                "handoff_reason": "low_confidence",
                "handoff_key": "handoff",
                "handoff_context": {
                    "best_score": round(best, 4),
                    "min_score": assistant.min_score,
                    "retrieved": retrieved_meta,
                },
                "agent_trace": ["retriever"],
            }

        return {
            "retrieved": results,
            "best_score": best,
            "attempts": 0,
            "grounding_issues": [],
            "agent_trace": ["retriever"],
        }

    def generator_agent(state: SanadGraphState) -> dict[str, Any]:
        eff_q = state["effective_query"]
        lang = state["language"]
        results = state.get("retrieved") or []
        issues = state.get("grounding_issues") or []
        attempts = int(state.get("attempts") or 0) + 1
        retrieved_meta = [source_from_result(r).model_dump() for r in results]

        try:
            messages = grounded_messages(
                eff_q,
                results,
                lang,
                retry_issues=issues if attempts > 1 else None,
            )
            reply = assistant.llm.chat(messages)
        except LLMError as exc:
            return {
                "attempts": attempts,
                "handoff_reason": "llm_error",
                "handoff_key": "handoff",
                "handoff_context": {"error": str(exc)},
                "agent_trace": ["generator"],
            }

        if is_handoff_reply(reply):
            return {
                "draft_answer": reply,
                "attempts": attempts,
                "handoff_reason": "model_uncertain",
                "handoff_key": "handoff",
                "handoff_context": {"retrieved": retrieved_meta},
                "agent_trace": ["generator"],
            }

        return {
            "draft_answer": reply,
            "attempts": attempts,
            "handoff_reason": None,
            "agent_trace": ["generator"],
        }

    def grounding_checker(state: SanadGraphState) -> dict[str, Any]:
        draft = state.get("draft_answer") or ""
        results = state.get("retrieved") or []
        lang = state["language"]
        cited, issues = check_grounding(
            state["effective_query"],
            draft,
            results,
            lang,
            check_numbers=assistant.strict_grounding,
            llm=assistant.llm,
            use_llm_judge=assistant.llm_grounding_check,
        )

        if not issues:
            return {
                "cited_indices": cited,
                "grounding_ok": True,
                "grounding_issues": [],
                "agent_trace": ["grounding_checker"],
            }

        max_attempts = int(state.get("max_attempts") or 1)
        attempts = int(state.get("attempts") or 1)
        if attempts < max_attempts:
            return {
                "cited_indices": cited,
                "grounding_ok": False,
                "grounding_issues": issues,
                "agent_trace": ["grounding_checker"],
            }

        reason = "no_citation" if issues == ["missing_citation"] else "grounding_failed"
        retrieved_meta = [source_from_result(r).model_dump() for r in results]
        return {
            "cited_indices": cited,
            "grounding_ok": False,
            "grounding_issues": issues,
            "handoff_reason": reason,
            "handoff_key": "handoff",
            "handoff_context": {
                "draft_answer": draft,
                "grounding_issues": issues,
                "retrieved": retrieved_meta,
            },
            "agent_trace": ["grounding_checker"],
        }

    def responder_node(state: SanadGraphState) -> dict[str, Any]:
        lang = state["language"]
        trace = list(state.get("agent_trace") or []) + ["responder"]

        if state.get("route") == "empty":
            resp = AssistantResponse(answer=msg("empty", lang), language=lang, trace=trace)
            return {"final_response": resp, "agent_trace": ["responder"]}

        results = state.get("retrieved") or []
        cited = state.get("cited_indices") or []
        answer = state.get("draft_answer") or ""
        resp = AssistantResponse(
            answer=answer,
            language=lang,
            sources=[source_from_result(results[n - 1]) for n in cited],
            trace=trace,
            grounding_issues=list(state.get("grounding_issues") or []),
        )
        return {
            "final_response": resp,
            "agent_trace": ["responder"],
            "history": [
                {"role": "user", "content": state["question"]},
                {"role": "assistant", "content": answer},
            ],
        }

    def handoff_node(state: SanadGraphState) -> dict[str, Any]:
        lang = state["language"]
        reason = state.get("handoff_reason") or "low_confidence"
        key = state.get("handoff_key") or "handoff"
        ctx = dict(state.get("handoff_context") or {})
        trace = list(state.get("agent_trace") or []) + ["handoff"]
        ctx["agent_trace"] = trace

        ticket = assistant.handoffs.create(
            state["question"],
            lang,
            reason,
            state.get("conversation_id"),
            ctx,
        )
        answer = msg(key, lang, ticket_id=ticket.id)
        resp = AssistantResponse(
            answer=answer,
            language=lang,
            handoff=Handoff(ticket_id=ticket.id, reason=reason),
            trace=trace,
            grounding_issues=list(state.get("grounding_issues") or []),
        )
        return {
            "final_response": resp,
            "agent_trace": ["handoff"],
            "history": [
                {"role": "user", "content": state["question"]},
                {"role": "assistant", "content": answer},
            ],
        }

    def _route_after_router(state: SanadGraphState) -> str:
        return state["route"]

    def _route_after_tool(state: SanadGraphState) -> str:
        return "done" if state.get("final_response") is not None else "rag"

    def _route_after_retriever(state: SanadGraphState) -> str:
        return "handoff" if state.get("handoff_reason") else "generate"

    def _route_after_generator(state: SanadGraphState) -> str:
        return "handoff" if state.get("handoff_reason") else "check"

    def _route_after_checker(state: SanadGraphState) -> str:
        if state.get("grounding_ok"):
            return "respond"
        if state.get("handoff_reason"):
            return "handoff"
        return "retry"

    builder = StateGraph(SanadGraphState)
    builder.add_node("router_agent", router_agent)
    builder.add_node("tool_agent", tool_agent)
    builder.add_node("retriever_agent", retriever_agent)
    builder.add_node("generator_agent", generator_agent)
    builder.add_node("grounding_checker", grounding_checker)
    builder.add_node("responder_node", responder_node)
    builder.add_node("handoff_node", handoff_node)

    builder.add_edge(START, "router_agent")
    builder.add_conditional_edges(
        "router_agent",
        _route_after_router,
        {
            "empty": "responder_node",
            "human": "handoff_node",
            "tool": "tool_agent",
            "rag": "retriever_agent",
        },
    )
    builder.add_conditional_edges(
        "tool_agent",
        _route_after_tool,
        {"done": END, "rag": "retriever_agent"},
    )
    builder.add_conditional_edges(
        "retriever_agent",
        _route_after_retriever,
        {"handoff": "handoff_node", "generate": "generator_agent"},
    )
    builder.add_conditional_edges(
        "generator_agent",
        _route_after_generator,
        {"handoff": "handoff_node", "check": "grounding_checker"},
    )
    builder.add_conditional_edges(
        "grounding_checker",
        _route_after_checker,
        {"respond": "responder_node", "retry": "generator_agent", "handoff": "handoff_node"},
    )
    builder.add_edge("responder_node", END)
    builder.add_edge("handoff_node", END)

    return builder.compile(checkpointer=InMemorySaver())


def invoke_sanad_graph(
    graph: Any,
    question: str,
    conversation_id: str | None = None,
    *,
    max_attempts: int = 1,
    callbacks: list[Any] | None = None,
) -> AssistantResponse:
    """Execute one conversational turn through the compiled LangGraph workflow."""
    thread_id = conversation_id or f"anon-{uuid.uuid4().hex}"
    initial_state: SanadGraphState = {
        "question": question,
        "conversation_id": conversation_id,
        "max_attempts": max_attempts,
        "handoff_reason": None,
        "handoff_context": {},
        "grounding_issues": [],
        "final_response": None,
        "agent_trace": ["__RESET__"],
    }
    run_config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if callbacks:
        run_config["callbacks"] = callbacks
    result = graph.invoke(initial_state, run_config)
    return result["final_response"]

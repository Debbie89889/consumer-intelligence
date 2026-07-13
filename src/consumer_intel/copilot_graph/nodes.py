"""Node functions for the customer-insight (chat) LangGraph graph.

    extract_context
      |-- customer_id still unresolved -> clarify -> END
      `-- resolved -> router
             |-- not found -> not_found -> END
             `-- exists -> fetch_rfm/clv/nbo/propensity -> join -> response_generator
                                                              |-- fallback -> fallback -> END
                                                              `-- normal -> END

Every fetch_* node calls an *existing* ``db/repository.py`` query function —
none of them re-implement SQL. ``fetch_rfm``/``fetch_clv``/``fetch_propensity``
all call the same :func:`repository.get_customer`: the analytics DB already
denormalises RFM, CLV and propensity into one row per customer (a Phase 5
design choice), so there is nothing to split them into. ``fetch_nbo`` is the
one node doing genuinely different work, via the
``next_best_offers_for_customer`` join query added alongside this graph.

No graph-level message persistence node: conversation history lives in the
SQLAlchemy ``messages`` table, managed entirely by ``chat.run_turn`` outside
the graph (see chat.py) — this graph has no checkpointer, so there is
nothing for a persistence node here to synchronise with.
"""

from __future__ import annotations

import os

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langgraph.graph import END
from sqlalchemy.orm import Session, sessionmaker

from consumer_intel.copilot.context import risk_from_prob_alive
from consumer_intel.copilot.narrator import get_chat_model
from consumer_intel.copilot_graph.chat_schema import ChatAnswer, ExtractedContext
from consumer_intel.copilot_graph.state import CopilotState
from consumer_intel.db import repository
from consumer_intel.labels import action_zh, risk_zh, segment_zh

CLARIFYING_QUESTION = "請問是想了解哪一位客戶?麻煩提供客戶編號。"

_EXTRACT_SYSTEM = (
    "你是一個 customer_id 解析器。根據對話歷史(可能包含代名詞,例如「他」「那位客戶」),"
    "判斷這一輪對話是在問哪一位客戶,customer_id 為純數字字串。"
    "如果對話中從未提及任何客戶編號,或無法判斷是哪一位,customer_id 請設為 null。"
    "不要猜測或捏造一個客戶編號。"
)

_ANSWER_SYSTEM = (
    "你是一位零售數據分析助理。你只能根據提供的『已計算好的事實』回答使用者的問題,"
    "不可以捏造或重新計算任何數字。若既有事實不足以回答,請誠實說明,不要編造。"
    "請用繁體中文回答。"
)


def _has_llm_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))


def extract_customer_id(messages: list[AnyMessage]) -> str | None:
    """LLM call: resolve which customer this turn is about, including pronouns."""
    model = get_chat_model()
    chain = model.with_structured_output(ExtractedContext)
    result: ExtractedContext = chain.invoke([SystemMessage(content=_EXTRACT_SYSTEM), *messages])
    return result.customer_id


def extract_context(state: CopilotState) -> dict:
    """Resolve which customer this turn is about, from conversation history.

    Skips the LLM call entirely when ``customer_id`` is already set (direct,
    non-chat invocations — tests, the benchmark script). No LLM key, or the
    call failing, means the reference genuinely can't be resolved here —
    falls through to the clarify path rather than crashing.
    """
    if state["customer_id"] is not None:
        return {}
    if not _has_llm_key():
        return {}
    try:
        customer_id = extract_customer_id(state["messages"])
    except Exception:
        customer_id = None
    return {"customer_id": customer_id}


def route_from_extract_context(state: CopilotState) -> str:
    return "clarify" if state["customer_id"] is None else "router"


def make_router(session_factory: sessionmaker[Session]):
    """Build the router node, bound to a session factory.

    Only decides ``customer_exists`` here (a cheap existence check) — the
    actual conditional routing logic lives in :func:`route_from_router`,
    which LangGraph runs against the state this node returns.
    """

    def router(state: CopilotState) -> dict:
        with session_factory() as session:
            exists = repository.customer_exists(session, str(state["customer_id"]))
        return {"customer_exists": exists}

    return router


def route_from_router(state: CopilotState) -> str | list[str]:
    if not state["customer_exists"]:
        return "not_found"
    return ["fetch_rfm", "fetch_clv", "fetch_nbo", "fetch_propensity"]


def not_found(state: CopilotState) -> dict:
    """Deterministic not-found response. Never reaches the LLM."""
    text = f"查無客戶 {state['customer_id']}。"
    return {"error": text, "messages": [AIMessage(content=text)]}


def clarify(state: CopilotState) -> dict:
    """Deterministic clarifying question when we can't tell which customer
    this turn is about. No LLM call."""
    return {
        "clarification": CLARIFYING_QUESTION,
        "messages": [AIMessage(content=CLARIFYING_QUESTION)],
    }


def make_fetch_rfm(session_factory: sessionmaker[Session]):
    """Build the fetch_rfm node, bound to a session factory."""

    def fetch_rfm(state: CopilotState) -> dict:
        with session_factory() as session:
            row = repository.get_customer(session, str(state["customer_id"]))
        return {"tool_results": {"rfm": row}}

    return fetch_rfm


def make_fetch_clv(session_factory: sessionmaker[Session]):
    """Build the fetch_clv node, bound to a session factory."""

    def fetch_clv(state: CopilotState) -> dict:
        with session_factory() as session:
            row = repository.get_customer(session, str(state["customer_id"]))
        return {"tool_results": {"clv": row}}

    return fetch_clv


def make_fetch_propensity(session_factory: sessionmaker[Session]):
    """Build the fetch_propensity node, bound to a session factory."""

    def fetch_propensity(state: CopilotState) -> dict:
        with session_factory() as session:
            row = repository.get_customer(session, str(state["customer_id"]))
        return {"tool_results": {"propensity": row}}

    return fetch_propensity


def make_fetch_nbo(session_factory: sessionmaker[Session]):
    """Build the fetch_nbo node, bound to a session factory."""

    def fetch_nbo(state: CopilotState) -> dict:
        with session_factory() as session:
            offers = repository.next_best_offers_for_customer(session, str(state["customer_id"]))
        return {"tool_results": {"nbo": offers}}

    return fetch_nbo


def join(state: CopilotState) -> dict:
    """Fan-in synchronisation point. router already ruled out a missing
    customer before the fan-out started, so there is nothing left to check
    here — this is a pass-through the graph needs so all four fetch edges
    converge before response_generator runs.
    """
    return {}


def build_chat_facts(customer_id: str, tool_results: dict) -> dict:
    """Flatten tool_results into the facts the chat LLM is allowed to see.

    Pure Python — risk_level, segment/action labels and offer names are all
    resolved here, not by the LLM. No per-customer facts flow to the LLM
    beyond what's returned here.
    """
    profile = tool_results.get("rfm") or {}
    offers = tool_results.get("nbo") or []
    prob_alive = float(profile.get("prob_alive") or 0.0)
    propensity = profile.get("propensity")
    return {
        "customer_id": customer_id,
        "segment": segment_zh(profile.get("segment")),
        "recency_days": profile.get("recency"),
        "frequency": profile.get("frequency"),
        "monetary": profile.get("monetary"),
        "predicted_clv": profile.get("predicted_clv"),
        "prob_alive": prob_alive,
        "risk_level": risk_zh(risk_from_prob_alive(prob_alive)),
        "propensity": float(propensity) if propensity is not None else None,
        "recommended_action": action_zh(profile.get("action")),
        "next_best_offers": [o["consequents"] for o in offers[:3]],
    }


def answer_template(facts: dict) -> str:
    """Deterministic fact summary (no LLM),繁體中文 — used when no key is
    set or the LLM call fails. Can't answer the specific question asked
    (that needs real language understanding), so it states the key facts
    instead of guessing."""
    parts = [
        f"客戶 {facts['customer_id']} 屬於「{facts['segment']}」客群,",
        f"最近一次購買在 {facts['recency_days']} 天前,",
        f"預估終身價值 £{(facts['predicted_clv'] or 0):,.0f},流失風險{facts['risk_level']}。",
    ]
    if facts["propensity"] is not None:
        parts.append(f"90 天回購傾向 {facts['propensity']:.0%}。")
    if facts["next_best_offers"]:
        parts.append("推薦商品:" + "、".join(facts["next_best_offers"]) + "。")
    return "".join(parts)


def answer_langchain(facts: dict, messages: list[AnyMessage]) -> str:
    """LLM-narrated answer to the user's actual latest question."""
    import json

    facts_json = json.dumps(facts, ensure_ascii=False)
    system = SystemMessage(content=f"{_ANSWER_SYSTEM}\n\n已計算好的事實(JSON):\n{facts_json}")
    model = get_chat_model()
    chain = model.with_structured_output(ChatAnswer)
    result: ChatAnswer = chain.invoke([system, *messages])
    return result.answer


def response_generator(state: CopilotState) -> dict:
    """Answer the user's latest question, grounded in the already-fetched facts.

    Falls back to a deterministic fact summary exactly like the plain LCEL
    Copilot's narrator does; records which backend served the answer so
    :func:`route_from_response_generator` can send a failed LLM call
    through the observable ``fallback`` node.
    """
    facts = build_chat_facts(str(state["customer_id"]), state["tool_results"])
    has_key = _has_llm_key()
    used = "langchain" if has_key else "template"
    try:
        answer = answer_langchain(facts, state["messages"]) if has_key else answer_template(facts)
    except Exception:
        answer = answer_template(facts)
        used = "template_fallback"
    return {
        "answer": answer,
        "narration_backend": used,
        "messages": [AIMessage(content=answer)],
    }


def route_from_response_generator(state: CopilotState) -> str:
    """Send a failed-LLM-fell-back-to-template result through the fallback node."""
    if state["narration_backend"] == "template_fallback":
        return "fallback"
    return END


def fallback(state: CopilotState) -> dict:
    """Observability marker for the (already-completed) fallback narration.

    ``response_generator`` already produced a valid template-based answer
    before this node runs — there is nothing left to compute. This node
    exists purely so the fallback path shows up as its own step in the
    graph and in execution traces, instead of being silently swallowed like
    it always has been for the plain LCEL Copilot.
    """
    return {}

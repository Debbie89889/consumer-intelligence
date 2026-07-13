"""Turns LangGraph ``astream_events()`` output into SSE-ready payloads.

Curates a small, JSON-serializable event schema instead of forwarding raw
LangChain event dicts — those carry full state snapshots and non-JSON
Python objects (``AIMessage``, etc.). Node start/end events are filtered to
this graph's own named nodes; internal sub-runnables (the prompt/model
chains inside ``extract_customer_id`` / ``answer_langchain``) are noise
here, not signal.

Known gap: ``on_chat_model_stream`` (the "LLM token" event type) is
implemented per LangChain's documented event schema, but has not been
empirically verified against a real provider in this environment — no
OPENAI_API_KEY/ANTHROPIC_API_KEY is available here to test it against a
live model. It's also worth knowing that both LLM calls in this graph use
``with_structured_output`` (tool-calling under the hood) for grounding, so
streamed deltas may be partial tool-call-argument fragments rather than
readable prose — a UI should treat "token" events as a progress signal, not
assume they concatenate into clean text (wait for the "final" event for
the validated answer).
"""

from __future__ import annotations

from consumer_intel.copilot_graph.chat import reply_text

NODE_NAMES = {
    "extract_context",
    "router",
    "clarify",
    "not_found",
    "fetch_rfm",
    "fetch_clv",
    "fetch_nbo",
    "fetch_propensity",
    "join",
    "response_generator",
    "fallback",
}


def sse_event_payload(event: dict) -> dict | None:
    """Return a JSON-ready dict for this LangChain event, or None to skip it."""
    kind = event["event"]
    name = event.get("name")

    if kind == "on_chain_start" and name in NODE_NAMES:
        return {"type": "node_start", "node": name}

    if kind == "on_chain_end" and name in NODE_NAMES:
        return {"type": "node_end", "node": name}

    if kind == "on_chain_end" and name == "LangGraph":
        output = event["data"].get("output") or {}
        payload = {"type": "final", "reply": reply_text(output)}
        if output.get("__interrupt__"):
            payload["type"] = "interrupt"
        return payload

    if kind == "on_chat_model_stream":
        chunk = event["data"].get("chunk")
        content = getattr(chunk, "content", None)
        return {"type": "token", "content": content} if content else None

    return None

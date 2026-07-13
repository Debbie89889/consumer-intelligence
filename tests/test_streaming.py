"""Tests for streaming.py's SSE event curation (pure function, no graph needed)."""

from __future__ import annotations

from consumer_intel.copilot_graph.streaming import sse_event_payload


def test_node_start_for_known_node():
    event = {"event": "on_chain_start", "name": "fetch_rfm", "data": {}}
    assert sse_event_payload(event) == {"type": "node_start", "node": "fetch_rfm"}


def test_node_end_for_known_node():
    event = {"event": "on_chain_end", "name": "response_generator", "data": {}}
    assert sse_event_payload(event) == {"type": "node_end", "node": "response_generator"}


def test_unknown_node_names_are_skipped():
    """Internal LangChain sub-runnables (prompt/model chains) aren't graph nodes."""
    event = {"event": "on_chain_start", "name": "ChatPromptTemplate", "data": {}}
    assert sse_event_payload(event) is None


def test_final_event_extracts_reply_text():
    event = {
        "event": "on_chain_end",
        "name": "LangGraph",
        "data": {"output": {"answer": "他是核心客戶。", "clarification": None, "error": None}},
    }
    assert sse_event_payload(event) == {"type": "final", "reply": "他是核心客戶。"}


def test_final_event_prefers_clarification_when_no_answer():
    event = {
        "event": "on_chain_end",
        "name": "LangGraph",
        "data": {"output": {"answer": None, "clarification": "請問是哪位客戶?", "error": None}},
    }
    assert sse_event_payload(event) == {"type": "final", "reply": "請問是哪位客戶?"}


def test_final_event_flags_interrupt():
    event = {
        "event": "on_chain_end",
        "name": "LangGraph",
        "data": {"output": {"answer": None, "__interrupt__": [object()]}},
    }
    result = sse_event_payload(event)
    assert result["type"] == "interrupt"


def test_chat_model_token_with_content():
    class FakeChunk:
        content = "測試"

    event = {"event": "on_chat_model_stream", "data": {"chunk": FakeChunk()}}
    assert sse_event_payload(event) == {"type": "token", "content": "測試"}


def test_chat_model_token_without_content_is_skipped():
    class FakeChunk:
        content = ""

    event = {"event": "on_chat_model_stream", "data": {"chunk": FakeChunk()}}
    assert sse_event_payload(event) is None


def test_other_event_types_are_skipped():
    event = {"event": "on_chain_stream", "name": "fetch_rfm", "data": {}}
    assert sse_event_payload(event) is None

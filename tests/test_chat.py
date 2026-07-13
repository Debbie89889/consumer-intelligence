"""Tests for chat.py (multi-turn conversation orchestration) and
extract_context's customer_id resolution.

Real pronoun resolution ("他"/"那" -> a specific customer_id) needs a real
LLM and isn't unit-testable; these tests verify the *wiring* instead — that
extract_context is actually given the accumulated conversation history
(not just the latest message) across turns, that run_turn correctly
loads/persists messages, and that whatever the (mocked) LLM resolves
correctly drives the rest of the graph.
"""

from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableLambda
from sqlalchemy.orm import sessionmaker

from consumer_intel.copilot_graph.chat import load_history, reply_text, run_turn
from consumer_intel.copilot_graph.chat_schema import ChatAnswer, ExtractedContext
from consumer_intel.copilot_graph.graph import build_customer_insight_graph


def _fake_chat_model(extracted_customer_id, answer_text="測試回答"):
    """A fake get_chat_model() result that dispatches by requested schema,
    so both extract_context's and response_generator's LLM calls work
    correctly through the same monkeypatch."""

    class FakeModel:
        def with_structured_output(self, schema):
            if schema is ExtractedContext:
                return RunnableLambda(
                    lambda _msgs: ExtractedContext(customer_id=extracted_customer_id)
                )
            return RunnableLambda(lambda _msgs: ChatAnswer(answer=answer_text))

    return FakeModel()


@pytest.fixture
def session_factory(populated_engine):
    return sessionmaker(bind=populated_engine)


@pytest.fixture
def graph(session_factory):
    return build_customer_insight_graph(session_factory)


def test_run_turn_resolves_and_answers_and_persists(graph, session_factory, monkeypatch):
    monkeypatch.setattr(
        "consumer_intel.copilot_graph.nodes.get_chat_model",
        lambda: _fake_chat_model("C1", "他是核心客戶。"),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-test")

    result = run_turn(graph, session_factory, "thread-1", "12345 這位客戶如何?")

    assert result["customer_id"] == "C1"
    assert result["answer"] == "他是核心客戶。"

    history = load_history(session_factory, "thread-1")
    assert len(history) == 2
    assert history[0].content == "12345 這位客戶如何?"
    assert history[0].type == "human"
    assert history[1].content == "他是核心客戶。"
    assert history[1].type == "ai"


def test_extract_context_sees_full_accumulated_history_across_turns(
    graph, session_factory, monkeypatch
):
    """The key pronoun-resolution wiring test: the second turn's extraction
    call must receive the first turn's messages too, not just the new one."""
    captured_message_counts = []

    def fake_get_chat_model():
        class FakeModel:
            def with_structured_output(self, schema):
                if schema is ExtractedContext:

                    def _invoke(msgs):
                        # msgs[0] is the system prompt; the rest is conversation history.
                        captured_message_counts.append(len(msgs) - 1)
                        return ExtractedContext(customer_id="C1")

                    return RunnableLambda(_invoke)
                return RunnableLambda(lambda _msgs: ChatAnswer(answer="回答"))

        return FakeModel()

    monkeypatch.setattr("consumer_intel.copilot_graph.nodes.get_chat_model", fake_get_chat_model)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-test")

    run_turn(graph, session_factory, "thread-2", "12345 這位客戶如何?")
    run_turn(graph, session_factory, "thread-2", "他為什麼被歸為 Champions?")
    run_turn(graph, session_factory, "thread-2", "那該給他什麼 offer?")

    # turn 1: just the new human message. turn 2: turn-1's human+AI + new
    # human. turn 3: all 4 prior + new human.
    assert captured_message_counts == [1, 3, 5]

    history = load_history(session_factory, "thread-2")
    assert len(history) == 6
    assert [m.type for m in history] == ["human", "ai", "human", "ai", "human", "ai"]
    assert history[2].content == "他為什麼被歸為 Champions?"
    assert history[4].content == "那該給他什麼 offer?"


def test_run_turn_not_found_persists_the_exchange(graph, session_factory, monkeypatch):
    monkeypatch.setattr(
        "consumer_intel.copilot_graph.nodes.get_chat_model",
        lambda: _fake_chat_model("NOPE"),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-test")

    result = run_turn(graph, session_factory, "thread-3", "99999 這位客戶如何?")
    assert result["error"] == "查無客戶 NOPE。"

    history = load_history(session_factory, "thread-3")
    assert len(history) == 2
    assert history[1].content == "查無客戶 NOPE。"


def test_run_turn_clarify_when_customer_unresolvable(graph, session_factory, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = run_turn(graph, session_factory, "thread-4", "你好")
    assert result["clarification"] is not None

    history = load_history(session_factory, "thread-4")
    assert len(history) == 2
    assert history[1].content == result["clarification"]


def test_reply_text_picks_whichever_field_is_set():
    assert reply_text({"answer": "a", "clarification": None, "error": None}) == "a"
    assert reply_text({"answer": None, "clarification": "c", "error": None}) == "c"
    assert reply_text({"answer": None, "clarification": None, "error": "e"}) == "e"
    assert reply_text({"answer": None, "clarification": None, "error": None}) == ""

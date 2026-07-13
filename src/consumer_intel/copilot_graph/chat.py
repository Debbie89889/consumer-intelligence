"""Orchestrates one conversation turn: load history, run the graph, persist new messages.

Conversation history lives in the SQLAlchemy ``messages`` table (business
data) — this graph has no checkpointer at all. Every turn reconstructs
history fresh from the ORM, appends the new human message, invokes the
graph, and persists whatever new messages the graph produced. Keeps the
checkpointer/ORM boundary crisp (see CLAUDE.md's "資料持久化分層" note):
there is no graph execution state to resume here, only a business-data
transcript to append to.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from sqlalchemy.orm import Session, sessionmaker

from consumer_intel.copilot_graph.state import CopilotState, initial_state
from consumer_intel.db.models import Conversation, Message


def _to_langchain_message(row: Message) -> AnyMessage:
    return (
        HumanMessage(content=row.content) if row.role == "human" else AIMessage(content=row.content)
    )


def load_history(session_factory: sessionmaker[Session], thread_id: str) -> list[AnyMessage]:
    """Reconstruct prior conversation turns as LangChain messages, oldest first."""
    with session_factory() as session:
        rows = (
            session.query(Message)
            .filter_by(thread_id=thread_id)
            .order_by(Message.created_at, Message.id)
            .all()
        )
        return [_to_langchain_message(r) for r in rows]


def persist_new_messages(
    session_factory: sessionmaker[Session],
    thread_id: str,
    customer_id: str | None,
    new_messages: list[AnyMessage],
) -> None:
    """Write this turn's new messages to the ORM, creating the Conversation row if needed."""
    if not new_messages:
        return
    with session_factory() as session:
        conversation = session.get(Conversation, thread_id)
        if conversation is None:
            session.add(Conversation(thread_id=thread_id, customer_id=customer_id))
        elif customer_id is not None:
            conversation.customer_id = customer_id
        for msg in new_messages:
            role = "human" if isinstance(msg, HumanMessage) else "ai"
            session.add(Message(thread_id=thread_id, role=role, content=msg.content))
        session.commit()


def reply_text(state: CopilotState) -> str:
    """The single text reply for this turn, whichever path produced it."""
    return state.get("answer") or state.get("clarification") or state.get("error") or ""


def run_turn(
    graph, session_factory: sessionmaker[Session], thread_id: str, user_text: str
) -> CopilotState:
    """Run one conversation turn: load history, invoke the graph, persist the new messages."""
    history = load_history(session_factory, thread_id)
    state_in = initial_state(
        thread_id, customer_id=None, messages=[*history, HumanMessage(content=user_text)]
    )
    result = graph.invoke(state_in)
    new_messages = result["messages"][len(history) :]
    persist_new_messages(session_factory, thread_id, result.get("customer_id"), new_messages)
    return result

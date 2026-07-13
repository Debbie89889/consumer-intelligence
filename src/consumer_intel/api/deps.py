"""FastAPI dependencies — database session and campaign-graph provisioning.

The session factory is created lazily from ``DATABASE_URL`` on first use. Tests
override :func:`get_db` via ``app.dependency_overrides`` to point at a
throwaway SQLite database, so no production connection is needed.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from consumer_intel.copilot_graph.campaign_graph import build_campaign_graph
from consumer_intel.copilot_graph.graph import build_customer_insight_graph
from consumer_intel.db.checkpointer import make_checkpointer
from consumer_intel.db.engine import make_session_factory

_session_factory: sessionmaker[Session] | None = None
_customer_insight_graph: Any | None = None


def get_session_factory() -> sessionmaker[Session]:
    """Lazily build (and cache) the session factory from the environment."""
    global _session_factory
    if _session_factory is None:
        _session_factory = make_session_factory()
    return _session_factory


def get_db() -> Iterator[Session]:
    """Yield a database session, closing it afterwards."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def get_campaign_graph() -> Iterator[Any]:
    """Yield a compiled campaign graph bound to a fresh checkpointer connection.

    Built per-request rather than once at app startup: SqliteSaver/PostgresSaver
    manage their own raw connection (not SQLAlchemy's pooled engine), so this
    trades a little per-request connection overhead for the same simple,
    dependency-injectable, test-overridable shape as :func:`get_db`.
    """
    with make_checkpointer() as checkpointer:
        yield build_campaign_graph(get_session_factory(), checkpointer)


def get_customer_insight_graph() -> Any:
    """Lazily build (and cache) the customer-insight chat graph.

    Unlike the campaign graph, this one has no checkpointer/per-request
    resource to manage — it's a stateless compiled execution plan bound to
    the (reusable) session factory, so a single cached instance is safe to
    reuse across requests.
    """
    global _customer_insight_graph
    if _customer_insight_graph is None:
        _customer_insight_graph = build_customer_insight_graph(get_session_factory())
    return _customer_insight_graph

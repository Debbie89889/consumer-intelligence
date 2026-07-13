"""LangGraph checkpointer factory — the execution-state counterpart to ``engine.py``.

Same URL-driven dispatch as :func:`consumer_intel.db.engine.make_engine`
(SQLite locally/in tests, PostgreSQL in production), but for a completely
different store: the checkpointer persists graph *execution* state (which
node ran, how to resume after an ``interrupt()``), never business data. See
the "資料持久化分層" note in CLAUDE.md.

Both ``SqliteSaver`` and ``PostgresSaver`` manage their own raw connection
(not a SQLAlchemy engine/session) and own internal tables, created via
``.setup()`` — entirely separate from the Alembic-managed ORM tables in
``db/models.py``.

Usage (context manager, mirroring the underlying savers' own API):

    with make_checkpointer() as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        ...
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from consumer_intel.db.engine import database_url


def _sqlite_path(url: str) -> str:
    """SQLAlchemy ``sqlite:///path`` -> a raw path for ``sqlite3.connect()``."""
    return url.removeprefix("sqlite:///")


def _psycopg_conn_string(url: str) -> str:
    """SQLAlchemy ``postgresql+psycopg2://...`` -> a plain psycopg3 conn string.

    langgraph-checkpoint-postgres uses psycopg (v3) directly, not SQLAlchemy —
    it doesn't understand the ``+psycopg2`` dialect suffix SQLAlchemy needs.
    """
    if "+psycopg2" in url:
        return url.replace("postgresql+psycopg2://", "postgresql://")
    return url


@contextmanager
def make_checkpointer(url: str | None = None) -> Iterator[object]:
    """Yield a set-up checkpointer for the given (or resolved) database URL."""
    resolved = url or database_url()
    if resolved.startswith("sqlite"):
        from langgraph.checkpoint.sqlite import SqliteSaver

        with SqliteSaver.from_conn_string(_sqlite_path(resolved)) as checkpointer:
            checkpointer.setup()
            yield checkpointer
    else:
        from langgraph.checkpoint.postgres import PostgresSaver

        with PostgresSaver.from_conn_string(_psycopg_conn_string(resolved)) as checkpointer:
            checkpointer.setup()
            yield checkpointer

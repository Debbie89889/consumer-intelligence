"""Database connection.

The same SQLAlchemy code targets PostgreSQL in production (via docker-compose)
and SQLite locally / in tests — chosen by the ``DATABASE_URL`` environment
variable. This is a common, pragmatic pattern: the application logic and SQL
stay identical; only the connection string changes.

    export DATABASE_URL=postgresql+psycopg2://intel:intel@localhost:5432/intel
    # default (no env set): a local SQLite file under data/
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from consumer_intel import config

DEFAULT_SQLITE_URL = f"sqlite:///{config.DATA_DIR / 'consumer_intel.db'}"


def _normalize_url(url: str) -> str:
    """Make a connection string SQLAlchemy-ready.

    Managed Postgres providers (Render, Heroku, ...) hand out ``postgres://``,
    but SQLAlchemy 2.x requires an explicit dialect+driver. Rewrite it to
    ``postgresql+psycopg2://`` so the same code works locally and in the cloud.
    """
    if url.startswith("postgres://"):
        return "postgresql+psycopg2://" + url[len("postgres://") :]
    return url


def database_url() -> str:
    """Resolve the database URL from the environment, defaulting to SQLite."""
    return _normalize_url(os.environ.get("DATABASE_URL", DEFAULT_SQLITE_URL))


def make_engine(url: str | None = None) -> Engine:
    """Create a SQLAlchemy engine for the given (or resolved) URL."""
    resolved = _normalize_url(url) if url else database_url()
    # check_same_thread is a SQLite-only concern (FastAPI uses threads).
    connect_args = {"check_same_thread": False} if resolved.startswith("sqlite") else {}
    return create_engine(resolved, connect_args=connect_args, future=True)


def make_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    """Build a session factory bound to an engine."""
    return sessionmaker(bind=engine or make_engine(), autoflush=False, future=True)


def get_session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """Yield a session and always close it (FastAPI dependency style)."""
    session = session_factory()
    try:
        yield session
    finally:
        session.close()

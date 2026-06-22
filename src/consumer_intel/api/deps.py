"""FastAPI dependencies — database session provisioning.

The session factory is created lazily from ``DATABASE_URL`` on first use. Tests
override :func:`get_db` via ``app.dependency_overrides`` to point at a
throwaway SQLite database, so no production connection is needed.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy.orm import Session, sessionmaker

from consumer_intel.db.engine import make_session_factory

_session_factory: sessionmaker[Session] | None = None


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

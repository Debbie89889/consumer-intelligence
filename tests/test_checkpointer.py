"""Tests for the LangGraph checkpointer factory (db/checkpointer.py).

Only the SQLite path is exercised end-to-end (no PostgreSQL available in
CI/local dev); the Postgres URL-translation helper is tested directly since
psycopg (v3)'s connection-string format differs from SQLAlchemy's
``+psycopg2`` dialect URL.
"""

from __future__ import annotations

from consumer_intel.db.checkpointer import _psycopg_conn_string, _sqlite_path, make_checkpointer


def test_sqlite_path_strips_sqlalchemy_prefix():
    assert _sqlite_path("sqlite:///data/consumer_intel.db") == "data/consumer_intel.db"


def test_psycopg_conn_string_strips_dialect_suffix():
    assert (
        _psycopg_conn_string("postgresql+psycopg2://u:p@host:5432/db")
        == "postgresql://u:p@host:5432/db"
    )


def test_psycopg_conn_string_passthrough_when_no_dialect_suffix():
    assert _psycopg_conn_string("postgresql://u:p@host:5432/db") == "postgresql://u:p@host:5432/db"


def test_make_checkpointer_sqlite_creates_usable_saver(tmp_path):
    with make_checkpointer(f"sqlite:///{tmp_path / 'cp.db'}") as checkpointer:
        # .setup() ran; the saver should be immediately usable (empty history).
        assert list(checkpointer.list(None)) == []

"""Tests for the Copilot business-data ORM models and their Alembic migration.

Model-level CRUD/constraint tests build the schema directly via
``Base.metadata.create_all`` (fast, standard for unit tests). A separate test
drives the actual Alembic migration end to end against a throwaway SQLite
database, so a broken migration script fails CI even though the ORM tests
above never touch ``alembic/``.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from consumer_intel.db.engine import make_engine
from consumer_intel.db.models import Base, CampaignApproval, Conversation, Message


@pytest.fixture
def engine(tmp_path):
    """A throwaway SQLite database with the Copilot business tables created directly."""
    eng = make_engine(f"sqlite:///{tmp_path / 'copilot.db'}")
    Base.metadata.create_all(eng)
    return eng


def test_conversation_message_crud(engine):
    with Session(engine) as s:
        convo = Conversation(thread_id="t1", customer_id="C1")
        s.add(convo)
        s.add(Message(thread_id="t1", role="human", content="12345 這位客戶如何？"))
        s.add(Message(thread_id="t1", role="ai", content="他是 Champions 客群。"))
        s.commit()

    with Session(engine) as s:
        convo = s.get(Conversation, "t1")
        assert convo is not None
        assert convo.customer_id == "C1"
        assert [m.role for m in convo.messages] == ["human", "ai"]


def test_message_tool_calls_json_roundtrip(engine):
    with Session(engine) as s:
        s.add(Conversation(thread_id="t1"))
        s.add(
            Message(
                thread_id="t1",
                role="tool",
                content="fetch_clv",
                tool_calls={"name": "fetch_clv", "args": {"customer_id": "C1"}},
            )
        )
        s.commit()

    with Session(engine) as s:
        msg = s.query(Message).filter_by(thread_id="t1").one()
        assert msg.tool_calls == {"name": "fetch_clv", "args": {"customer_id": "C1"}}


def test_message_rejects_invalid_role(engine):
    with Session(engine) as s:
        s.add(Conversation(thread_id="t1"))
        s.add(Message(thread_id="t1", role="system", content="not allowed"))
        with pytest.raises(IntegrityError):
            s.commit()


def test_campaign_approval_defaults_to_pending(engine):
    with Session(engine) as s:
        s.add(Conversation(thread_id="t1"))
        s.add(
            CampaignApproval(
                thread_id="t1",
                draft={"customers": ["C1", "C2"], "offer": "20725", "discount": 0.1},
            )
        )
        s.commit()

    with Session(engine) as s:
        approval = s.query(CampaignApproval).filter_by(thread_id="t1").one()
        assert approval.status == "pending"
        assert approval.draft["customers"] == ["C1", "C2"]


def test_campaign_approval_rejects_invalid_status(engine):
    with Session(engine) as s:
        s.add(Conversation(thread_id="t1"))
        s.add(CampaignApproval(thread_id="t1", draft={}, status="approved_by_mistake"))
        with pytest.raises(IntegrityError):
            s.commit()


def test_deleting_conversation_cascades_to_children(engine):
    with Session(engine) as s:
        s.add(Conversation(thread_id="t1"))
        s.add(Message(thread_id="t1", role="human", content="hi"))
        s.add(CampaignApproval(thread_id="t1", draft={}))
        s.commit()

    with Session(engine) as s:
        s.delete(s.get(Conversation, "t1"))
        s.commit()

    with Session(engine) as s:
        assert s.query(Message).count() == 0
        assert s.query(CampaignApproval).count() == 0


def test_alembic_migration_upgrade_and_downgrade(tmp_path, monkeypatch):
    """Drive the real migration script (not create_all) against a fresh SQLite db."""
    from alembic.config import Config

    from alembic import command
    from consumer_intel import config as app_config

    db_path = tmp_path / "migrated.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path.as_posix()}")

    alembic_cfg = Config(str(app_config.PROJECT_ROOT / "alembic.ini"))

    command.upgrade(alembic_cfg, "head")
    eng = make_engine(os.environ["DATABASE_URL"])
    tables = set(inspect(eng).get_table_names())
    assert {"conversations", "messages", "campaign_approvals"} <= tables
    eng.dispose()

    command.downgrade(alembic_cfg, "base")
    eng = make_engine(os.environ["DATABASE_URL"])
    tables = set(inspect(eng).get_table_names())
    assert not {"conversations", "messages", "campaign_approvals"} & tables
    eng.dispose()

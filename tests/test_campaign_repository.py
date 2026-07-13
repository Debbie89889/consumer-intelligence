"""Tests for db/campaign_repository.py (read queries against campaign_approvals)."""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from consumer_intel.db import campaign_repository
from consumer_intel.db.engine import make_engine
from consumer_intel.db.models import Base, CampaignApproval, Conversation


@pytest.fixture
def engine(tmp_path):
    eng = make_engine(f"sqlite:///{tmp_path / 'campaigns.db'}")
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        s.add(Conversation(thread_id="t1"))
        s.add(Conversation(thread_id="t2"))
        s.add(
            CampaignApproval(
                thread_id="t1",
                draft={"brief": {"headline": "h1"}, "candidates": []},
                status="pending",
            )
        )
        s.add(
            CampaignApproval(
                thread_id="t2",
                draft={"brief": {"headline": "h2"}, "candidates": []},
                status="approved",
                reviewer="alice",
            )
        )
        s.commit()
    return eng


def test_list_campaigns_all(engine):
    rows = campaign_repository.list_campaigns(Session(engine))
    assert {r["thread_id"] for r in rows} == {"t1", "t2"}


def test_list_campaigns_filtered_by_status(engine):
    rows = campaign_repository.list_campaigns(Session(engine), status="pending")
    assert [r["thread_id"] for r in rows] == ["t1"]


def test_get_campaign_found(engine):
    row = campaign_repository.get_campaign(Session(engine), "t2")
    assert row is not None
    assert row["status"] == "approved"
    assert row["reviewer"] == "alice"
    assert row["draft"]["brief"]["headline"] == "h2"


def test_get_campaign_unknown(engine):
    assert campaign_repository.get_campaign(Session(engine), "nope") is None

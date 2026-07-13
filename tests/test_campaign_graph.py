"""End-to-end tests for the win-back campaign HITL graph: interrupt, resume
(approved / revised / rejected), and the campaign_approvals DB writes at
each step.

Uses a dedicated fixture (not the shared ``populated_engine`` from
conftest.py) because this needs both a purpose-built win-back candidate set
and the ORM tables (``conversations``/``campaign_approvals``) that graph
writes to, which ``populated_engine`` doesn't create.
"""

from __future__ import annotations

import pandas as pd
import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from sqlalchemy.orm import sessionmaker

from consumer_intel.copilot_graph.campaign_graph import build_campaign_graph
from consumer_intel.copilot_graph.campaign_state import initial_campaign_state
from consumer_intel.db.models import Base, CampaignApproval

WIN_BACK_IDS = {"W1", "W2", "W3", "W4", "W5", "W6"}


@pytest.fixture
def campaign_engine(tmp_path):
    """SQLite db with 6 win-back candidates (2 per discount tier), one
    non-candidate (Champions, must be excluded), a rule for W1's top
    product, and no rule for W2's — plus the ORM tables the graph writes to.
    """
    from consumer_intel.db.engine import make_engine

    engine = make_engine(f"sqlite:///{tmp_path / 'campaign.db'}")
    customers = pd.DataFrame(
        [
            {
                "customer_id": "W1",
                "segment": "At Risk",
                "recency": 200,
                "frequency": 5,
                "monetary": 3000.0,
                "predicted_clv": 900.0,
                "prob_alive": 0.35,
            },
            {
                "customer_id": "W2",
                "segment": "At Risk",
                "recency": 180,
                "frequency": 4,
                "monetary": 2500.0,
                "predicted_clv": 700.0,
                "prob_alive": 0.40,
            },
            {
                "customer_id": "W3",
                "segment": "Can't Lose Them",
                "recency": 250,
                "frequency": 8,
                "monetary": 6000.0,
                "predicted_clv": 500.0,
                "prob_alive": 0.20,
            },
            {
                "customer_id": "W4",
                "segment": "At Risk",
                "recency": 150,
                "frequency": 3,
                "monetary": 1500.0,
                "predicted_clv": 300.0,
                "prob_alive": 0.45,
            },
            {
                "customer_id": "W5",
                "segment": "Can't Lose Them",
                "recency": 300,
                "frequency": 6,
                "monetary": 4000.0,
                "predicted_clv": 200.0,
                "prob_alive": 0.10,
            },
            {
                "customer_id": "W6",
                "segment": "At Risk",
                "recency": 220,
                "frequency": 2,
                "monetary": 800.0,
                "predicted_clv": 100.0,
                "prob_alive": 0.50,
            },
            {
                "customer_id": "N1",
                "segment": "Champions",
                "recency": 5,
                "frequency": 20,
                "monetary": 10000.0,
                "predicted_clv": 2000.0,
                "prob_alive": 0.99,
            },
        ]
    )
    customers.to_sql("customers", engine, index=False, if_exists="replace")

    rules = pd.DataFrame(
        [
            {
                "antecedents": "LUNCH BAG RED",
                "consequents": "LUNCH BAG PINK",
                "antecedents_codes": "20725",
                "consequents_codes": "22384",
                "support": 0.02,
                "confidence": 0.18,
                "lift": 9.9,
            }
        ]
    )
    rules.to_sql("rules", engine, index=False, if_exists="replace")

    customer_top_product = pd.DataFrame(
        [{"customer_id": "W1", "stock_code": "20725", "revenue": 500.0}]
    )
    customer_top_product.to_sql("customer_top_product", engine, index=False, if_exists="replace")

    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def session_factory(campaign_engine):
    return sessionmaker(bind=campaign_engine)


@pytest.fixture
def graph_and_config(session_factory, tmp_path):
    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoints.db")) as checkpointer:
        checkpointer.setup()
        graph = build_campaign_graph(session_factory, checkpointer)
        config = {"configurable": {"thread_id": "campaign-thread-1"}}
        yield graph, config


def _thread_id(config):
    return config["configurable"]["thread_id"]


def test_first_invoke_pauses_at_await_approval_with_pending_row(graph_and_config, session_factory):
    graph, config = graph_and_config
    result = graph.invoke(initial_campaign_state(_thread_id(config)), config=config)

    assert "__interrupt__" in result
    assert graph.get_state(config).next == ("await_approval",)

    with session_factory() as s:
        approval = s.query(CampaignApproval).filter_by(thread_id=_thread_id(config)).one()
        assert approval.status == "pending"
        assert approval.draft["brief"] is not None
        candidate_ids = {c["customer_id"] for c in approval.draft["candidates"]}
        assert candidate_ids == WIN_BACK_IDS  # N1 (Champions) excluded


def test_match_offers_attaches_nbo_where_a_rule_exists(graph_and_config):
    graph, config = graph_and_config
    graph.invoke(initial_campaign_state(_thread_id(config)), config=config)

    candidates = graph.get_state(config).values["candidates"]
    w1 = next(c for c in candidates if c["customer_id"] == "W1")
    w2 = next(c for c in candidates if c["customer_id"] == "W2")
    assert w1["offer"] == "LUNCH BAG PINK"
    assert w2["offer"] is None  # no top-product rule for W2


def test_discount_tiers_assigned_across_all_six_candidates(graph_and_config):
    graph, config = graph_and_config
    graph.invoke(initial_campaign_state(_thread_id(config)), config=config)

    by_id = {c["customer_id"]: c["discount"] for c in graph.get_state(config).values["candidates"]}
    assert by_id["W1"] == 0.20 and by_id["W2"] == 0.20  # top tier (900, 700)
    assert by_id["W3"] == 0.15 and by_id["W4"] == 0.15  # mid tier (500, 300)
    assert by_id["W5"] == 0.10 and by_id["W6"] == 0.10  # bottom tier (200, 100)


def test_resume_approved_commits_and_writes_reviewer(graph_and_config, session_factory):
    graph, config = graph_and_config
    graph.invoke(initial_campaign_state(_thread_id(config)), config=config)

    result = graph.invoke(
        Command(resume={"action": "approved", "reviewer": "alice"}), config=config
    )
    assert result["status"] == "approved"

    with session_factory() as s:
        approval = s.query(CampaignApproval).filter_by(thread_id=_thread_id(config)).one()
        assert approval.status == "approved"
        assert approval.reviewer == "alice"
        assert approval.decided_at is not None


def test_resume_revised_loops_back_to_draft_with_edits_applied(graph_and_config, session_factory):
    graph, config = graph_and_config
    graph.invoke(initial_campaign_state(_thread_id(config)), config=config)

    result = graph.invoke(
        Command(
            resume={
                "action": "revised",
                "review_note": "折扣太深,W6 剔除",
                "excluded_customer_ids": ["W6"],
                "discount_overrides": {"W1": 0.12},
            }
        ),
        config=config,
    )

    # paused again — a second interrupt after the redraft
    assert "__interrupt__" in result
    assert graph.get_state(config).next == ("await_approval",)

    with session_factory() as s:
        approval = s.query(CampaignApproval).filter_by(thread_id=_thread_id(config)).one()
        assert approval.status == "pending"  # still pending, now with a new draft
        ids = {c["customer_id"] for c in approval.draft["candidates"]}
        assert "W6" not in ids
        w1 = next(c for c in approval.draft["candidates"] if c["customer_id"] == "W1")
        assert w1["discount"] == 0.12
        # W1's earlier-matched offer must survive the redraft (match_offers isn't rerun)
        assert w1["offer"] == "LUNCH BAG PINK"


def test_resume_revised_then_approved_commits_final_edited_list(graph_and_config, session_factory):
    graph, config = graph_and_config
    graph.invoke(initial_campaign_state(_thread_id(config)), config=config)
    graph.invoke(
        Command(resume={"action": "revised", "excluded_customer_ids": ["W6"]}), config=config
    )
    result = graph.invoke(
        Command(resume={"action": "approved", "reviewer": "carol"}), config=config
    )

    assert result["status"] == "approved"
    with session_factory() as s:
        approval = s.query(CampaignApproval).filter_by(thread_id=_thread_id(config)).one()
        assert approval.status == "approved"
        ids = {c["customer_id"] for c in approval.draft["candidates"]}
        assert ids == WIN_BACK_IDS - {"W6"}


def test_resume_rejected_marks_db_rejected(graph_and_config, session_factory):
    graph, config = graph_and_config
    graph.invoke(initial_campaign_state(_thread_id(config)), config=config)

    result = graph.invoke(
        Command(resume={"action": "rejected", "review_note": "不做這次活動", "reviewer": "bob"}),
        config=config,
    )
    assert result["status"] == "rejected"

    with session_factory() as s:
        approval = s.query(CampaignApproval).filter_by(thread_id=_thread_id(config)).one()
        assert approval.status == "rejected"
        assert approval.reviewer == "bob"
        assert approval.decided_at is not None

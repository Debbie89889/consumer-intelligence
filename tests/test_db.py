"""Tests for the database repository queries (run against SQLite)."""

from __future__ import annotations

from sqlalchemy.orm import Session

from consumer_intel.db import repository


def test_count_customers(populated_engine):
    with Session(populated_engine) as s:
        assert repository.count_customers(s) == 3


def test_postgres_url_is_normalized_for_sqlalchemy():
    from consumer_intel.db.engine import _normalize_url

    # Render/Heroku style -> SQLAlchemy dialect+driver
    assert _normalize_url("postgres://u:p@host:5432/db") == (
        "postgresql+psycopg2://u:p@host:5432/db"
    )
    # already-correct and sqlite URLs pass through unchanged
    assert _normalize_url("postgresql+psycopg2://x") == "postgresql+psycopg2://x"
    assert _normalize_url("sqlite:///x.db") == "sqlite:///x.db"


def test_get_customer_found(populated_engine):
    with Session(populated_engine) as s:
        row = repository.get_customer(s, "C1")
    assert row is not None
    assert row["segment"] == "Champions"
    assert row["predicted_clv"] == 1200.0


def test_get_customer_missing(populated_engine):
    with Session(populated_engine) as s:
        assert repository.get_customer(s, "NOPE") is None


def test_segment_summary_aggregates(populated_engine):
    with Session(populated_engine) as s:
        summary = repository.segment_summary(s)
    by_seg = {r["segment"]: r for r in summary}
    assert by_seg["Champions"]["customers"] == 1
    # ordered by total_revenue desc
    revenues = [r["total_revenue"] for r in summary]
    assert revenues == sorted(revenues, reverse=True)


def test_top_customers_by_clv_orders_desc(populated_engine):
    with Session(populated_engine) as s:
        top = repository.top_customers_by_clv(s, limit=2)
    assert len(top) == 2
    assert top[0]["customer_id"] == "C1"  # highest predicted_clv
    assert top[0]["predicted_clv"] >= top[1]["predicted_clv"]


def test_next_best_offers_by_product(populated_engine):
    with Session(populated_engine) as s:
        recs = repository.next_best_offers(s, "20725", limit=5)
    assert len(recs) == 2
    # ranked by lift desc
    assert recs[0]["lift"] >= recs[1]["lift"]
    assert recs[0]["consequents_codes"] == "22384"


def test_next_best_offers_unknown_product(populated_engine):
    with Session(populated_engine) as s:
        assert repository.next_best_offers(s, "99999") == []

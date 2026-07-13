"""Tests for the database repository queries (run against SQLite)."""

from __future__ import annotations

from sqlalchemy.orm import Session

from consumer_intel.db import repository


def test_win_back_candidates_filters_by_segment(populated_engine):
    with Session(populated_engine) as s:
        rows = repository.win_back_candidates(s)
    # only C2 ("Can't Lose Them") qualifies; C1 (Champions) and C3 (Lost) don't
    assert [r["customer_id"] for r in rows] == ["C2"]
    assert rows[0]["segment"] == "Can't Lose Them"


def test_customer_exists_true_for_known_customer(populated_engine):
    with Session(populated_engine) as s:
        assert repository.customer_exists(s, "C1") is True


def test_customer_exists_false_for_unknown_customer(populated_engine):
    with Session(populated_engine) as s:
        assert repository.customer_exists(s, "NOPE") is False


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


def test_next_best_offers_for_customer(populated_engine):
    with Session(populated_engine) as s:
        recs = repository.next_best_offers_for_customer(s, "C1", limit=5)
    assert len(recs) == 2
    assert recs[0]["lift"] >= recs[1]["lift"]
    assert recs[0]["consequents_codes"] == "22384"


def test_next_best_offers_for_customer_no_matching_rules(populated_engine):
    with Session(populated_engine) as s:
        assert repository.next_best_offers_for_customer(s, "C2") == []


def test_next_best_offers_for_customer_unknown_customer(populated_engine):
    with Session(populated_engine) as s:
        assert repository.next_best_offers_for_customer(s, "NOPE") == []


def test_list_customers_browse(populated_engine):
    with Session(populated_engine) as s:
        rows = repository.list_customers(s, limit=10)
    assert len(rows) == 3
    # highest spend first (C2 monetary 8000)
    assert rows[0]["customer_id"] == "C2"
    assert "customer_id" in rows[0]


def test_top_products_orders_by_revenue(populated_engine):
    with Session(populated_engine) as s:
        rows = repository.top_products(s, limit=2)
    assert len(rows) == 2
    assert rows[0]["stock_code"] == "85123A"  # highest revenue
    assert rows[0]["revenue"] >= rows[1]["revenue"]


def test_get_product(populated_engine):
    with Session(populated_engine) as s:
        prod = repository.get_product(s, "20725")
        missing = repository.get_product(s, "NOPE")
    assert prod is not None
    assert prod["description"] == "LUNCH BAG RED"
    assert missing is None


def test_monthly_series_chronological(populated_engine):
    with Session(populated_engine) as s:
        rows = repository.monthly_series(s)
    assert [r["month"] for r in rows] == ["2010-01", "2010-02", "2010-03"]


def test_country_summary_orders_by_revenue(populated_engine):
    with Session(populated_engine) as s:
        rows = repository.country_summary(s, limit=2)
    assert len(rows) == 2
    assert rows[0]["country"] == "United Kingdom"
    assert rows[0]["revenue"] >= rows[1]["revenue"]


def test_product_overview(populated_engine):
    with Session(populated_engine) as s:
        ov = repository.product_overview(s)
    assert ov["products"] == 3
    assert ov["revenue"] == 42000.0
    assert ov["quantity"] == 17000

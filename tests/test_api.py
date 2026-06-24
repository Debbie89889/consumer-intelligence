"""Tests for the FastAPI service, using a throwaway SQLite database."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from consumer_intel.api.app import app
from consumer_intel.api.deps import get_db


@pytest.fixture
def client(populated_engine):
    """TestClient with get_db overridden to use the populated SQLite engine."""

    def _override():
        s = Session(populated_engine)
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["customers"] == 3


def test_get_customer(client):
    r = client.get("/customers/C1")
    assert r.status_code == 200
    body = r.json()
    assert body["segment"] == "Champions"
    assert body["predicted_clv"] == 1200.0


def test_get_customer_404(client):
    r = client.get("/customers/UNKNOWN")
    assert r.status_code == 404


def test_segments(client):
    r = client.get("/segments")
    assert r.status_code == 200
    segs = {row["segment"] for row in r.json()}
    assert {"Champions", "Can't Lose Them", "Lost"}.issubset(segs)


def test_top_clv(client):
    r = client.get("/customers/top-clv", params={"limit": 2})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 2
    assert rows[0]["customer_id"] == "C1"


def test_next_best_offer(client):
    r = client.get("/products/20725/next-best-offer", params={"limit": 5})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 2
    assert rows[0]["lift"] >= rows[1]["lift"]


def test_customer_insight_is_grounded(client):
    r = client.get("/customers/C2/insight")
    assert r.status_code == 200
    body = r.json()
    # C2 has prob_alive 0.30 -> high churn risk (computed in Python, not the LLM)
    assert body["risk_level"] == "high"
    assert body["segment"] == "Can't Lose Them"
    assert len(body["observations"]) >= 1
    assert body["grounding"]["predicted_clv"] == 300.0


def test_top_clv_limit_validation(client):
    # limit above the allowed maximum -> 422 from FastAPI query validation
    r = client.get("/customers/top-clv", params={"limit": 9999})
    assert r.status_code == 422


def test_customers_browse(client):
    r = client.get("/customers", params={"limit": 10})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 3
    assert rows[0]["customer_id"] == "C2"  # highest monetary


def test_products_list(client):
    r = client.get("/products", params={"limit": 2})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 2
    assert rows[0]["stock_code"] == "85123A"


def test_product_detail(client):
    r = client.get("/products/20725")
    assert r.status_code == 200
    assert r.json()["description"] == "LUNCH BAG RED"


def test_product_detail_404(client):
    r = client.get("/products/NOPE")
    assert r.status_code == 404


def test_analytics_monthly(client):
    r = client.get("/analytics/monthly")
    assert r.status_code == 200
    months = [row["month"] for row in r.json()]
    assert months == ["2010-01", "2010-02", "2010-03"]


def test_analytics_countries(client):
    r = client.get("/analytics/countries", params={"limit": 2})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 2
    assert rows[0]["country"] == "United Kingdom"


def test_products_overview(client):
    r = client.get("/analytics/products-overview")
    assert r.status_code == 200
    body = r.json()
    assert body["products"] == 3
    assert body["revenue"] == 42000.0

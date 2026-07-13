"""Tests for the FastAPI service, using a throwaway SQLite database."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from consumer_intel.api.app import app
from consumer_intel.api.deps import get_campaign_graph, get_customer_insight_graph, get_db


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


@pytest.fixture
def campaign_client(populated_engine, tmp_path):
    """TestClient with get_db AND get_campaign_graph both overridden.

    populated_engine already has C2 ("Can't Lose Them") as a win-back
    candidate and the ORM tables the campaign graph writes to.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    from consumer_intel.copilot_graph.campaign_graph import build_campaign_graph

    def _override_db():
        s = Session(populated_engine)
        try:
            yield s
        finally:
            s.close()

    session_factory = sessionmaker(bind=populated_engine)

    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoints.db")) as checkpointer:
        checkpointer.setup()
        graph = build_campaign_graph(session_factory, checkpointer)

        def _override_graph():
            yield graph

        app.dependency_overrides[get_db] = _override_db
        app.dependency_overrides[get_campaign_graph] = _override_graph
        with TestClient(app) as c:
            yield c
        app.dependency_overrides.clear()


@pytest.fixture
def chat_client(populated_engine):
    """TestClient with get_db AND get_customer_insight_graph both overridden."""
    from consumer_intel.copilot_graph.graph import build_customer_insight_graph

    def _override_db():
        s = Session(populated_engine)
        try:
            yield s
        finally:
            s.close()

    graph = build_customer_insight_graph(sessionmaker(bind=populated_engine))

    def _override_graph():
        return graph

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_customer_insight_graph] = _override_graph
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _sse_events(response):
    events = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


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


# --- campaign endpoints ----------------------------------------------------


def test_generate_campaign_pauses_for_review(campaign_client):
    r = campaign_client.post("/campaigns/generate")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "pending_review"
    assert body["brief"]["headline"]
    assert {c["customer_id"] for c in body["candidates"]} == {"C2"}  # only win-back candidate


def test_list_campaigns_after_generate(campaign_client):
    gen = campaign_client.post("/campaigns/generate").json()
    r = campaign_client.get("/campaigns", params={"status": "pending"})
    assert r.status_code == 200
    threads = {c["thread_id"] for c in r.json()}
    assert gen["thread_id"] in threads


def test_get_campaign_detail(campaign_client):
    gen = campaign_client.post("/campaigns/generate").json()
    r = campaign_client.get(f"/campaigns/{gen['thread_id']}")
    assert r.status_code == 200
    assert r.json()["candidates"][0]["customer_id"] == "C2"


def test_get_campaign_detail_unknown(campaign_client):
    r = campaign_client.get("/campaigns/nope-not-a-thread")
    assert r.status_code == 404


def test_resume_approved_commits(campaign_client):
    gen = campaign_client.post("/campaigns/generate").json()
    r = campaign_client.post(
        f"/campaigns/{gen['thread_id']}/resume",
        json={"action": "approved", "reviewer": "alice"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "approved"
    assert body["reviewer"] == "alice"

    detail = campaign_client.get(f"/campaigns/{gen['thread_id']}").json()
    assert detail["status"] == "approved"
    assert detail["decided_at"] is not None


def test_resume_revised_pauses_again_with_edits(campaign_client):
    gen = campaign_client.post("/campaigns/generate").json()
    r = campaign_client.post(
        f"/campaigns/{gen['thread_id']}/resume",
        json={
            "action": "revised",
            "review_note": "調整折扣",
            "discount_overrides": {"C2": 0.05},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "pending_review"
    assert body["candidates"][0]["discount"] == 0.05


def test_resume_rejected(campaign_client):
    gen = campaign_client.post("/campaigns/generate").json()
    r = campaign_client.post(
        f"/campaigns/{gen['thread_id']}/resume",
        json={"action": "rejected", "reviewer": "bob", "review_note": "不執行"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "rejected"


def test_resume_unknown_thread_404(campaign_client):
    r = campaign_client.post("/campaigns/nope-not-a-thread/resume", json={"action": "approved"})
    assert r.status_code == 404


def test_resume_already_decided_400(campaign_client):
    gen = campaign_client.post("/campaigns/generate").json()
    campaign_client.post(f"/campaigns/{gen['thread_id']}/resume", json={"action": "approved"})
    r = campaign_client.post(f"/campaigns/{gen['thread_id']}/resume", json={"action": "approved"})
    assert r.status_code == 400


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


# --- chat streaming endpoint -------------------------------------------


def test_chat_stream_clarify_when_no_llm_key(chat_client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with chat_client.stream(
        "GET", "/chat/stream", params={"thread_id": "t1", "message": "你好"}
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = _sse_events(r)

    node_names = [e["node"] for e in events if e["type"] in ("node_start", "node_end")]
    assert "extract_context" in node_names
    assert "clarify" in node_names
    assert "router" not in node_names  # never reached — customer_id unresolved

    final = events[-1]
    assert final["type"] == "final"
    assert final["reply"] == "請問是想了解哪一位客戶?麻煩提供客戶編號。"


def test_chat_stream_answers_and_persists(chat_client, monkeypatch):
    from langchain_core.runnables import RunnableLambda

    from consumer_intel.copilot_graph.chat_schema import ChatAnswer, ExtractedContext

    class FakeModel:
        def with_structured_output(self, schema):
            if schema is ExtractedContext:
                return RunnableLambda(lambda _msgs: ExtractedContext(customer_id="C1"))
            return RunnableLambda(lambda _msgs: ChatAnswer(answer="他是核心客戶。"))

    monkeypatch.setattr("consumer_intel.copilot_graph.nodes.get_chat_model", lambda: FakeModel())
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-test")

    with chat_client.stream(
        "GET", "/chat/stream", params={"thread_id": "t2", "message": "C1 這位客戶如何?"}
    ) as r:
        assert r.status_code == 200
        events = _sse_events(r)

    node_names = [e["node"] for e in events if e["type"] in ("node_start", "node_end")]
    assert "fetch_rfm" in node_names
    assert "response_generator" in node_names

    final = events[-1]
    assert final["type"] == "final"
    assert final["reply"] == "他是核心客戶。"

    # a second turn on the same thread should see the first turn's history
    with chat_client.stream(
        "GET", "/chat/stream", params={"thread_id": "t2", "message": "他為什麼被歸為 Champions?"}
    ) as r2:
        events2 = _sse_events(r2)
    assert events2[-1]["type"] == "final"


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

"""FastAPI service exposing the consumer-intelligence outputs.

Endpoints (see interactive docs at ``/docs``):

* ``GET  /health``                              — service + row count
* ``GET  /customers/{id}``                      — full RFM/CLV/propensity profile
* ``GET  /customers/{id}/insight``              — grounded LLM insight (Copilot)
* ``GET  /segments``                            — per-segment rollup
* ``GET  /customers/top-clv``                   — highest predicted-CLV customers
* ``GET  /products/{stock_code}/next-best-offer`` — cross-sell recommendations
* ``POST /campaigns/generate``                  — draft a win-back campaign (HITL)
* ``GET  /campaigns``                           — list campaign drafts
* ``GET  /campaigns/{thread_id}``                — one campaign draft's detail
* ``POST /campaigns/{thread_id}/resume``         — human review decision (approve/revise/reject)
* ``GET  /chat/stream``                          — conversational customer Q&A, streamed via SSE

All numbers come from SQL/Python; the Copilot only narrates them.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from sqlalchemy.orm import Session

from consumer_intel.api import models
from consumer_intel.api.deps import (
    get_campaign_graph,
    get_customer_insight_graph,
    get_db,
    get_session_factory,
)
from consumer_intel.copilot.context import build_context
from consumer_intel.copilot.narrator import generate_insight
from consumer_intel.copilot.schema import CustomerInsight
from consumer_intel.copilot_graph.campaign_state import initial_campaign_state
from consumer_intel.copilot_graph.chat import load_history, persist_new_messages
from consumer_intel.copilot_graph.state import initial_state
from consumer_intel.copilot_graph.streaming import sse_event_payload
from consumer_intel.db import campaign_repository, repository

app = FastAPI(
    title="Consumer Intelligence API",
    version="0.1.0",
    description="Serves customer segments, CLV, propensity and Next Best Offer.",
)


@app.get("/health", response_model=models.HealthResponse)
def health(db: Session = Depends(get_db)) -> models.HealthResponse:
    return models.HealthResponse(status="ok", customers=repository.count_customers(db))


@app.get("/customers", response_model=list[models.CustomerListItem])
def customers(
    limit: int = Query(100, ge=1, le=1000), db: Session = Depends(get_db)
) -> list[models.CustomerListItem]:
    return [models.CustomerListItem(**r) for r in repository.list_customers(db, limit)]


@app.get("/customers/top-clv", response_model=list[models.TopCustomer])
def top_clv(
    limit: int = Query(20, ge=1, le=200), db: Session = Depends(get_db)
) -> list[models.TopCustomer]:
    return [models.TopCustomer(**r) for r in repository.top_customers_by_clv(db, limit)]


@app.get("/customers/{customer_id}", response_model=models.CustomerProfile)
def customer(customer_id: str, db: Session = Depends(get_db)) -> models.CustomerProfile:
    row = repository.get_customer(db, customer_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    return models.CustomerProfile(**row)


@app.get("/customers/{customer_id}/insight", response_model=CustomerInsight)
def customer_insight(customer_id: str, db: Session = Depends(get_db)) -> CustomerInsight:
    row = repository.get_customer(db, customer_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    ctx = build_context(row)
    return generate_insight(ctx)  # backend="auto": LLM if a key is set, else template


@app.get("/segments", response_model=list[models.SegmentSummaryItem])
def segments(db: Session = Depends(get_db)) -> list[models.SegmentSummaryItem]:
    return [models.SegmentSummaryItem(**r) for r in repository.segment_summary(db)]


@app.get("/analytics/monthly", response_model=list[models.MonthlyPoint])
def analytics_monthly(db: Session = Depends(get_db)) -> list[models.MonthlyPoint]:
    return [models.MonthlyPoint(**r) for r in repository.monthly_series(db)]


@app.get("/analytics/countries", response_model=list[models.CountrySummary])
def analytics_countries(
    limit: int = Query(15, ge=1, le=100), db: Session = Depends(get_db)
) -> list[models.CountrySummary]:
    return [models.CountrySummary(**r) for r in repository.country_summary(db, limit)]


@app.get("/analytics/products-overview", response_model=models.ProductsOverview)
def analytics_products_overview(db: Session = Depends(get_db)) -> models.ProductsOverview:
    return models.ProductsOverview(**repository.product_overview(db))


@app.get("/products", response_model=list[models.ProductSummary])
def products(
    limit: int = Query(20, ge=1, le=500), db: Session = Depends(get_db)
) -> list[models.ProductSummary]:
    return [models.ProductSummary(**r) for r in repository.top_products(db, limit)]


@app.get("/products/{stock_code}", response_model=models.ProductSummary)
def product(stock_code: str, db: Session = Depends(get_db)) -> models.ProductSummary:
    row = repository.get_product(db, stock_code)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Product {stock_code} not found")
    return models.ProductSummary(**row)


@app.get(
    "/products/{stock_code}/next-best-offer",
    response_model=list[models.NextBestOffer],
)
def next_best_offer(
    stock_code: str, limit: int = Query(5, ge=1, le=20), db: Session = Depends(get_db)
) -> list[models.NextBestOffer]:
    return [models.NextBestOffer(**r) for r in repository.next_best_offers(db, stock_code, limit)]


def _campaign_draft_from_state(thread_id: str, values: dict) -> models.CampaignDraft:
    return models.CampaignDraft(
        thread_id=thread_id,
        status=values.get("status", "unknown"),
        brief=values.get("brief"),
        candidates=values.get("candidates") or [],
        reviewer=values.get("reviewer"),
        review_note=values.get("review_note"),
    )


@app.post("/campaigns/generate", response_model=models.CampaignDraft)
def generate_campaign(graph: Any = Depends(get_campaign_graph)) -> models.CampaignDraft:
    """Draft a win-back campaign for the At Risk / Can't Lose Them segments.

    Runs the graph up to its interrupt() — a human must review via
    ``/campaigns/{thread_id}/resume`` before anything is committed.
    """
    state = initial_campaign_state()
    config = {"configurable": {"thread_id": state["thread_id"]}}
    graph.invoke(state, config=config)
    snapshot = graph.get_state(config)
    return _campaign_draft_from_state(state["thread_id"], snapshot.values)


@app.get("/campaigns", response_model=list[models.CampaignSummaryItem])
def list_campaigns(
    status: str | None = Query(None), db: Session = Depends(get_db)
) -> list[models.CampaignSummaryItem]:
    rows = campaign_repository.list_campaigns(db, status=status)
    return [
        models.CampaignSummaryItem(
            thread_id=r["thread_id"],
            status=r["status"],
            headline=(r["draft"] or {}).get("brief", {}).get("headline"),
            candidate_count=len((r["draft"] or {}).get("candidates") or []),
            created_at=r["created_at"],
        )
        for r in rows
    ]


@app.get("/campaigns/{thread_id}", response_model=models.CampaignDraft)
def get_campaign(thread_id: str, db: Session = Depends(get_db)) -> models.CampaignDraft:
    row = campaign_repository.get_campaign(db, thread_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Campaign {thread_id} not found")
    draft = row["draft"] or {}
    return models.CampaignDraft(
        thread_id=row["thread_id"],
        status=row["status"],
        brief=draft.get("brief"),
        candidates=draft.get("candidates") or [],
        reviewer=row["reviewer"],
        review_note=row["review_note"],
        decided_at=row["decided_at"],
        created_at=row["created_at"],
    )


@app.post("/campaigns/{thread_id}/resume", response_model=models.CampaignDraft)
def resume_campaign(
    thread_id: str,
    body: models.CampaignResumeRequest,
    graph: Any = Depends(get_campaign_graph),
) -> models.CampaignDraft:
    """Apply a human review decision: approved / revised / rejected.

    A "revised" decision loops the graph back to re-draft and pauses again
    (a second interrupt); the response reflects whatever state the graph is
    in after this call, which may again be "pending_review".
    """
    config = {"configurable": {"thread_id": thread_id}}
    if not graph.get_state(config).values:
        raise HTTPException(status_code=404, detail=f"Campaign {thread_id} not found")
    if not graph.get_state(config).next:
        raise HTTPException(status_code=400, detail=f"Campaign {thread_id} is not awaiting review")

    graph.invoke(Command(resume=body.model_dump()), config=config)
    snapshot = graph.get_state(config)
    return _campaign_draft_from_state(thread_id, snapshot.values)


@app.get("/chat/stream")
async def chat_stream(
    thread_id: str,
    message: str,
    graph: Any = Depends(get_customer_insight_graph),
) -> StreamingResponse:
    """Conversational customer Q&A, streamed as SSE.

    Resolves customer_id (including pronouns like "他") from the thread's
    conversation history via the graph's extract_context node; the new
    human/AI messages are persisted once the run completes. Event payloads
    are curated by streaming.sse_event_payload — see that module for the
    event schema and a known gap around LLM token-level streaming.
    """
    session_factory = get_session_factory()
    history = load_history(session_factory, thread_id)
    state_in = initial_state(
        thread_id, customer_id=None, messages=[*history, HumanMessage(content=message)]
    )

    async def event_generator():
        final_state: dict | None = None
        async for event in graph.astream_events(state_in, version="v2"):
            if event["event"] == "on_chain_end" and event.get("name") == "LangGraph":
                final_state = event["data"].get("output")
            payload = sse_event_payload(event)
            if payload is not None:
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        if final_state is not None:
            new_messages = final_state["messages"][len(history) :]
            persist_new_messages(
                session_factory, thread_id, final_state.get("customer_id"), new_messages
            )

    return StreamingResponse(event_generator(), media_type="text/event-stream")

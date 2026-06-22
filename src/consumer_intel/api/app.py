"""FastAPI service exposing the consumer-intelligence outputs.

Endpoints (see interactive docs at ``/docs``):

* ``GET  /health``                              — service + row count
* ``GET  /customers/{id}``                      — full RFM/CLV/propensity profile
* ``GET  /customers/{id}/insight``              — grounded LLM insight (Copilot)
* ``GET  /segments``                            — per-segment rollup
* ``GET  /customers/top-clv``                   — highest predicted-CLV customers
* ``GET  /products/{stock_code}/next-best-offer`` — cross-sell recommendations

All numbers come from SQL/Python; the Copilot only narrates them.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy.orm import Session

from consumer_intel.api import models
from consumer_intel.api.deps import get_db
from consumer_intel.copilot.context import build_context
from consumer_intel.copilot.narrator import generate_insight
from consumer_intel.copilot.schema import CustomerInsight
from consumer_intel.db import repository

app = FastAPI(
    title="Consumer Intelligence API",
    version="0.1.0",
    description="Serves customer segments, CLV, propensity and Next Best Offer.",
)


@app.get("/health", response_model=models.HealthResponse)
def health(db: Session = Depends(get_db)) -> models.HealthResponse:
    return models.HealthResponse(status="ok", customers=repository.count_customers(db))


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


@app.get(
    "/products/{stock_code}/next-best-offer",
    response_model=list[models.NextBestOffer],
)
def next_best_offer(
    stock_code: str, limit: int = Query(5, ge=1, le=20), db: Session = Depends(get_db)
) -> list[models.NextBestOffer]:
    return [models.NextBestOffer(**r) for r in repository.next_best_offers(db, stock_code, limit)]

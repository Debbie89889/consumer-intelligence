"""Pydantic response models for the API (the public data contract)."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    customers: int


class CustomerProfile(BaseModel):
    customer_id: str
    recency: int | None = None
    frequency: int | None = None
    monetary: float | None = None
    rfm_score: str | None = None
    segment: str | None = None
    action: str | None = None
    cluster_name: str | None = None
    predicted_clv: float | None = None
    prob_alive: float | None = None
    predicted_purchases: float | None = None
    historical_clv: float | None = None
    clv_method: str | None = None
    propensity: float | None = None


class SegmentSummaryItem(BaseModel):
    segment: str | None = None
    customers: int
    total_revenue: float | None = None
    avg_monetary: float | None = None
    avg_predicted_clv: float | None = None
    avg_prob_alive: float | None = None


class TopCustomer(BaseModel):
    customer_id: str
    segment: str | None = None
    predicted_clv: float | None = None
    prob_alive: float | None = None
    propensity: float | None = None


class NextBestOffer(BaseModel):
    antecedents: str
    consequents: str
    consequents_codes: str
    support: float
    confidence: float
    lift: float


class CustomerListItem(BaseModel):
    customer_id: str
    segment: str | None = None
    recency: int | None = None
    frequency: int | None = None
    monetary: float | None = None
    predicted_clv: float | None = None
    propensity: float | None = None


class ProductSummary(BaseModel):
    stock_code: str
    description: str | None = None
    revenue: float | None = None
    quantity: int | None = None
    orders: int | None = None
    customers: int | None = None


class MonthlyPoint(BaseModel):
    month: str
    revenue: float | None = None
    orders: int | None = None
    customers: int | None = None


class CountrySummary(BaseModel):
    country: str
    revenue: float | None = None
    orders: int | None = None
    customers: int | None = None


class ProductsOverview(BaseModel):
    products: int
    revenue: float | None = None
    quantity: int | None = None

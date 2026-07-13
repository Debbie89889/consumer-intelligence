"""Pydantic response models for the API (the public data contract)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


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


class CampaignCandidateItem(BaseModel):
    customer_id: str
    segment: str | None = None
    recency: int | None = None
    frequency: int | None = None
    monetary: float | None = None
    predicted_clv: float | None = None
    prob_alive: float | None = None
    discount: float | None = None
    offer: str | None = None


class CampaignBriefItem(BaseModel):
    headline: str
    message: str
    selling_points: list[str]


class CampaignDraft(BaseModel):
    thread_id: str
    status: str
    brief: CampaignBriefItem | None = None
    candidates: list[CampaignCandidateItem] = Field(default_factory=list)
    reviewer: str | None = None
    review_note: str | None = None
    decided_at: datetime | None = None
    created_at: datetime | None = None


class CampaignSummaryItem(BaseModel):
    thread_id: str
    status: str
    headline: str | None = None
    candidate_count: int
    created_at: datetime | None = None


class CampaignResumeRequest(BaseModel):
    action: Literal["approved", "revised", "rejected"]
    reviewer: str | None = None
    review_note: str | None = None
    excluded_customer_ids: list[str] = Field(default_factory=list)
    discount_overrides: dict[str, float] = Field(default_factory=dict)

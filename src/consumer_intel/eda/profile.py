"""Exploratory summaries over the cleaned transactions table.

Every function returns a DataFrame (or small dict) — no plotting, no printing —
so the numbers are reusable and testable. Plotting lives in the Phase 0 script.

All functions assume the canonical cleaned schema:
``InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice,
CustomerID, Country, TotalPrice``.
"""

from __future__ import annotations

import pandas as pd


def dataset_overview(df: pd.DataFrame) -> dict[str, object]:
    """Headline counts: rows, customers, invoices, products, date span, revenue."""
    return {
        "n_line_items": len(df),
        "n_customers": df["CustomerID"].nunique(),
        "n_invoices": df["InvoiceNo"].nunique(),
        "n_products": df["StockCode"].nunique(),
        "n_countries": df["Country"].nunique(),
        "date_min": df["InvoiceDate"].min(),
        "date_max": df["InvoiceDate"].max(),
        "total_revenue": float(df["TotalPrice"].sum()),
    }


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per calendar month: revenue, orders, active customers, average order value."""
    g = df.assign(_month=df["InvoiceDate"].dt.to_period("M").dt.to_timestamp())
    by_month = (
        g.groupby("_month")
        .agg(
            revenue=("TotalPrice", "sum"),
            orders=("InvoiceNo", "nunique"),
            customers=("CustomerID", "nunique"),
        )
        .reset_index()
        .rename(columns={"_month": "month"})
    )
    by_month["avg_order_value"] = by_month["revenue"] / by_month["orders"]
    return by_month.sort_values("month").reset_index(drop=True)


def country_summary(df: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
    """Revenue, orders and customers by country, descending by revenue."""
    by_country = (
        df.groupby("Country")
        .agg(
            revenue=("TotalPrice", "sum"),
            orders=("InvoiceNo", "nunique"),
            customers=("CustomerID", "nunique"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
    by_country["revenue_share"] = by_country["revenue"] / by_country["revenue"].sum()
    return by_country.head(top_n) if top_n else by_country


def order_value_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of order (invoice) totals: count, mean, median, quantiles."""
    order_totals = df.groupby("InvoiceNo")["TotalPrice"].sum()
    desc = order_totals.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99])
    return desc.rename("order_value").reset_index().rename(columns={"index": "stat"})


def _modal_description(s: pd.Series) -> str:
    """Most common non-null description for a product (stable label)."""
    non_null = s.dropna()
    return non_null.mode().iloc[0] if len(non_null) else ""


def top_products(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Top ``n`` products by revenue, with units sold and a description label."""
    by_product = (
        df.groupby("StockCode")
        .agg(
            revenue=("TotalPrice", "sum"),
            units=("Quantity", "sum"),
            description=("Description", _modal_description),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    return by_product


def customer_value_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Per-customer total spend distribution (motivates RFM/CLV in later phases)."""
    spend = df.groupby("CustomerID")["TotalPrice"].sum()
    desc = spend.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99])
    return desc.rename("customer_spend").reset_index().rename(columns={"index": "stat"})

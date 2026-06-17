"""Predictive CLV with BG/NBD (purchase count) + Gamma-Gamma (spend).

Two probabilistic models, fitted with the ``lifetimes`` library:

* **BG/NBD** models how many future purchases each customer will make and
  the probability they are still "alive" (active). Valid for *all* customers,
  including one-time buyers.
* **Gamma-Gamma** models the average monetary value per transaction. It needs
  *repeat* purchases to identify a customer's spend distribution, so it is
  fitted only on customers with ``frequency > 0`` and positive monetary value.

For one-time buyers we cannot estimate an individual spend distribution, so
their CLV uses a documented fallback: BG/NBD expected purchases times the
population mean transaction value. Every row is tagged with ``clv_method`` so
the two populations are never silently mixed.

Note on assumptions: Gamma-Gamma assumes purchase frequency and monetary value
are uncorrelated; :func:`frequency_monetary_correlation` reports that check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

# Average days per month, used to convert a horizon in months to the day units
# of the recency/T columns when asking BG/NBD for expected purchases.
DAYS_PER_MONTH = 365.25 / 12
DEFAULT_PENALIZER = 0.001


@dataclass
class CLVResult:
    """Fitted models plus the per-customer prediction table."""

    summary: pd.DataFrame
    bgf: BetaGeoFitter
    ggf: GammaGammaFitter
    predictions: pd.DataFrame
    horizon_months: int


def build_summary(
    transactions: pd.DataFrame,
    observation_end: pd.Timestamp | None = None,
    freq: str = "D",
) -> pd.DataFrame:
    """Build the lifetimes RFM summary (frequency, recency, T, monetary_value).

    Note these use the *lifetimes* definitions, which differ from the marketing
    RFM in :mod:`consumer_intel.features.rfm`:
    ``frequency`` counts *repeat* purchases (first purchase excluded);
    ``recency`` is the age at last purchase; ``T`` is the customer's total age;
    ``monetary_value`` is the mean value of *repeat* transactions.
    """
    if observation_end is None:
        observation_end = transactions["InvoiceDate"].max()
    return summary_data_from_transaction_data(
        transactions,
        customer_id_col="CustomerID",
        datetime_col="InvoiceDate",
        monetary_value_col="TotalPrice",
        observation_period_end=observation_end,
        freq=freq,
    )


def eligible_mask(summary: pd.DataFrame) -> pd.Series:
    """Customers usable by Gamma-Gamma: repeat buyers with positive spend."""
    return (summary["frequency"] > 0) & (summary["monetary_value"] > 0)


def frequency_monetary_correlation(summary: pd.DataFrame) -> float:
    """Pearson correlation of frequency and monetary value on eligible customers.

    Gamma-Gamma assumes this is close to zero.
    """
    rep = summary[eligible_mask(summary)]
    return float(rep["frequency"].corr(rep["monetary_value"]))


def fit_bgnbd(summary: pd.DataFrame, penalizer: float = DEFAULT_PENALIZER) -> BetaGeoFitter:
    """Fit the BG/NBD model on all customers."""
    bgf = BetaGeoFitter(penalizer_coef=penalizer)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])
    return bgf


def fit_gamma_gamma(
    summary: pd.DataFrame, penalizer: float = DEFAULT_PENALIZER
) -> GammaGammaFitter:
    """Fit the Gamma-Gamma model on eligible (repeat, positive-spend) customers."""
    rep = summary[eligible_mask(summary)]
    ggf = GammaGammaFitter(penalizer_coef=penalizer)
    ggf.fit(rep["frequency"], rep["monetary_value"])
    return ggf


def predict(
    summary: pd.DataFrame,
    bgf: BetaGeoFitter,
    ggf: GammaGammaFitter,
    horizon_months: int = 3,
    discount_rate: float = 0.01,
    freq: str = "D",
) -> pd.DataFrame:
    """Per-customer predictions over ``horizon_months``.

    Returns a frame indexed by ``CustomerID`` with:
    ``predicted_purchases``, ``prob_alive``, ``predicted_avg_value``,
    ``predicted_clv`` and ``clv_method``.
    """
    horizon_days = horizon_months * DAYS_PER_MONTH
    elig = eligible_mask(summary)

    out = pd.DataFrame(index=summary.index)

    # lifetimes' BG/NBD formula takes log(hyp2f1(...)); for rare numerically
    # extreme customers hyp2f1 can return a non-positive value, so numpy emits
    # "invalid value encountered in log" and yields NaN. We knowingly handle
    # that NaN in the defensive guards below, so we silence the (benign) FP
    # warning at its source rather than letting it surface to callers/CI.
    with np.errstate(invalid="ignore", divide="ignore"):
        # BG/NBD — valid for everyone (incl. one-time buyers).
        out["predicted_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
            horizon_days, summary["frequency"], summary["recency"], summary["T"]
        )
        out["prob_alive"] = bgf.conditional_probability_alive(
            summary["frequency"], summary["recency"], summary["T"]
        )

        # Gamma-Gamma average value — only meaningful for eligible customers.
        rep = summary[elig]
        avg_value = ggf.conditional_expected_average_profit(rep["frequency"], rep["monetary_value"])
        pop_avg_value = float(avg_value.mean())

        out["predicted_avg_value"] = pop_avg_value  # default = population mean
        out.loc[elig, "predicted_avg_value"] = avg_value

        # CLV: proper discounted model for eligible; documented fallback otherwise.
        clv_elig = ggf.customer_lifetime_value(
            bgf,
            rep["frequency"],
            rep["recency"],
            rep["T"],
            rep["monetary_value"],
            time=horizon_months,
            freq=freq,
            discount_rate=discount_rate,
        )
    out["clv_method"] = "fallback_pop_mean"
    out.loc[elig, "clv_method"] = "bg_nbd_gamma_gamma"
    out["predicted_clv"] = out["predicted_purchases"] * pop_avg_value  # fallback
    out.loc[elig, "predicted_clv"] = clv_elig

    # Defensive guards: lifetimes can emit NaN for rare numerically-extreme
    # customers and tiny negative floating-point values. A CLV table feeding a
    # dashboard must not contain NaN/negatives, so floor them to 0 (no expected
    # value) and backfill avg value with the population mean.
    out["predicted_avg_value"] = out["predicted_avg_value"].fillna(pop_avg_value)
    out["predicted_purchases"] = out["predicted_purchases"].fillna(0.0).clip(lower=0)
    out["predicted_clv"] = out["predicted_clv"].fillna(0.0).clip(lower=0)

    return out


def compute_predictive_clv(
    transactions: pd.DataFrame,
    horizon_months: int = 3,
    observation_end: pd.Timestamp | None = None,
    discount_rate: float = 0.01,
    penalizer: float = DEFAULT_PENALIZER,
    freq: str = "D",
) -> CLVResult:
    """End-to-end: build summary, fit both models, predict CLV.

    Returns a :class:`CLVResult` bundling the summary, fitted models and the
    per-customer prediction table.
    """
    summary = build_summary(transactions, observation_end=observation_end, freq=freq)
    bgf = fit_bgnbd(summary, penalizer=penalizer)
    ggf = fit_gamma_gamma(summary, penalizer=penalizer)
    preds = predict(
        summary,
        bgf,
        ggf,
        horizon_months=horizon_months,
        discount_rate=discount_rate,
        freq=freq,
    )
    return CLVResult(
        summary=summary,
        bgf=bgf,
        ggf=ggf,
        predictions=preds,
        horizon_months=horizon_months,
    )

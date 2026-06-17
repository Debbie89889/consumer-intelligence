"""Calibration / holdout validation for the BG/NBD purchase model.

Fitting a model is not evidence it predicts. We split each customer's history
at a cut-off date: fit BG/NBD on the *calibration* window, predict expected
purchases over the *holdout* window, then compare to what actually happened.
Reported metrics (MAE, RMSE, correlation, aggregate predicted vs actual) show
whether the model generalises rather than just fits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes.utils import calibration_and_holdout_data

from consumer_intel.clv.predictive import DEFAULT_PENALIZER


@dataclass
class HoldoutValidation:
    """Per-customer predicted vs actual holdout purchases, plus summary metrics."""

    data: pd.DataFrame  # columns include predicted_purchases, frequency_holdout
    metrics: dict[str, float]


def calibration_holdout(
    transactions: pd.DataFrame,
    calibration_end: pd.Timestamp,
    observation_end: pd.Timestamp,
    freq: str = "D",
    penalizer: float = DEFAULT_PENALIZER,
) -> HoldoutValidation:
    """Validate BG/NBD on a calibration/holdout split.

    Parameters
    ----------
    calibration_end:
        End of the calibration (training) window.
    observation_end:
        End of the holdout (test) window — usually the last date in the data.
    """
    cal_holdout = calibration_and_holdout_data(
        transactions,
        customer_id_col="CustomerID",
        datetime_col="InvoiceDate",
        calibration_period_end=calibration_end,
        observation_period_end=observation_end,
        freq=freq,
    )

    bgf = BetaGeoFitter(penalizer_coef=penalizer)
    bgf.fit(
        cal_holdout["frequency_cal"],
        cal_holdout["recency_cal"],
        cal_holdout["T_cal"],
    )

    holdout_days = (observation_end - calibration_end).days
    predicted = bgf.predict(
        holdout_days,
        cal_holdout["frequency_cal"],
        cal_holdout["recency_cal"],
        cal_holdout["T_cal"],
    )
    actual = cal_holdout["frequency_holdout"]

    out = cal_holdout.copy()
    out["predicted_purchases"] = predicted

    err = predicted - actual
    metrics = {
        "n_customers": int(len(out)),
        "holdout_days": int(holdout_days),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "correlation": float(np.corrcoef(predicted, actual)[0, 1]),
        "actual_total_purchases": float(actual.sum()),
        "predicted_total_purchases": float(predicted.sum()),
    }
    return HoldoutValidation(data=out, metrics=metrics)

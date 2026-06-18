"""Propensity models: a logistic-regression baseline and a LightGBM model,
plus evaluation (ROC-AUC, PR-AUC, Brier) and calibration.

The baseline exists so the tree model has to *earn* its place: if LightGBM
doesn't beat a scaled logistic regression, the extra complexity isn't worth it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from consumer_intel.propensity.features import FEATURE_COLUMNS, LABEL_COLUMN

RANDOM_STATE = 42


def split_xy(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix and the binary label."""
    return table[FEATURE_COLUMNS], table[LABEL_COLUMN]


def train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split (preserves the positive rate)."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def fit_logistic(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Baseline: standardise features then fit logistic regression."""
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def fit_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
    **params: object,
) -> LGBMClassifier:
    """Fit a LightGBM classifier (sensible defaults, overridable via params)."""
    defaults = dict(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        verbose=-1,
    )
    defaults.update(params)
    model = LGBMClassifier(**defaults)
    model.fit(X_train, y_train)
    return model


def predict_proba(model: object, X: pd.DataFrame) -> np.ndarray:
    """Positive-class probability for any fitted sklearn-style classifier."""
    return model.predict_proba(X)[:, 1]


def evaluate(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """ROC-AUC, PR-AUC, Brier score and the test base rate."""
    p = predict_proba(model, X_test)
    return {
        "roc_auc": float(roc_auc_score(y_test, p)),
        "pr_auc": float(average_precision_score(y_test, p)),
        "brier": float(brier_score_loss(y_test, p)),
        "base_rate": float(y_test.mean()),
        "n_test": int(len(y_test)),
    }


def calibration_table(
    model: object, X_test: pd.DataFrame, y_test: pd.Series, n_bins: int = 10
) -> pd.DataFrame:
    """Reliability table: mean predicted probability vs observed frequency."""
    p = predict_proba(model, X_test)
    frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean_pred, "fraction_positive": frac_pos})

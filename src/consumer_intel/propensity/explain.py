"""SHAP-based explanation for the tree model.

Global importance = mean absolute SHAP value per feature on a sample of rows.
SHAP attributes each prediction back to its features, so this answers "what
drives the propensity score" in a way plain split-count importance can't.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import shap


def shap_importance(model: object, X: pd.DataFrame, max_samples: int = 2000) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, sorted descending.

    Parameters
    ----------
    model:
        A fitted tree model (e.g. LightGBM) compatible with ``shap.TreeExplainer``.
    X:
        Feature matrix to explain. Sampled down to ``max_samples`` rows for speed.

    Returns
    -------
    DataFrame with columns ``feature`` and ``mean_abs_shap``.
    """
    sample = X.sample(min(len(X), max_samples), random_state=0) if len(X) > max_samples else X

    explainer = shap.TreeExplainer(model)
    # shap warns that LightGBM binary output is a list of ndarrays; we handle
    # exactly that below, so the notice is noise — silence it at the source.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*output has changed to a list.*")
        values = explainer.shap_values(sample)
    # Binary classifiers may return a list [class0, class1]; take the positive
    # class. Newer versions return a single array already.
    if isinstance(values, list):
        values = values[1]

    mean_abs = np.abs(values).mean(axis=0)
    return (
        pd.DataFrame({"feature": sample.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

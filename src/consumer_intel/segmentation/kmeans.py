"""K-means clustering over RFM as a data-driven complement to the rule grid.

Pipeline: log-transform the skewed RFM columns, standardise, then K-means.
``evaluate_k_range`` produces elbow (inertia) and silhouette curves to choose
``k``; ``fit_kmeans`` fits the final model; ``profile_clusters`` /
``name_clusters`` turn opaque cluster ids into readable customer profiles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
RFM_COLUMNS = ["Recency", "Frequency", "Monetary"]


@dataclass
class ClusterModel:
    """A fitted K-means model plus the scaler used to build its features."""

    kmeans: KMeans
    scaler: StandardScaler
    k: int


def prepare_features(rfm: pd.DataFrame) -> np.ndarray:
    """Log-transform (``log1p``) and standardise the RFM columns.

    Frequency and Monetary are strongly right-skewed and Recency mildly so;
    ``log1p`` keeps the big wholesale accounts from dominating the distance
    metric. Standardising puts the three features on a comparable scale so no
    single one drives the Euclidean distance.
    """
    logged = np.log1p(rfm[RFM_COLUMNS].to_numpy())
    return StandardScaler().fit_transform(logged)


def _fit_scaler_and_transform(rfm: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    logged = np.log1p(rfm[RFM_COLUMNS].to_numpy())
    scaler = StandardScaler().fit(logged)
    return scaler.transform(logged), scaler


def evaluate_k_range(
    rfm: pd.DataFrame,
    k_values: range | list[int] | None = None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """For each k: inertia (elbow) and mean silhouette score.

    Returns a DataFrame with columns ``k``, ``inertia``, ``silhouette``.
    ``k_values`` defaults to ``range(2, 11)``.
    """
    if k_values is None:
        k_values = range(2, 11)
    X, _ = _fit_scaler_and_transform(rfm)
    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if k > 1 else float("nan")
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def choose_k(metrics: pd.DataFrame, k_min: int = 3, k_max: int = 8) -> int:
    """Pick k by best silhouette within ``[k_min, k_max]``.

    Silhouette frequently peaks at k=2, which is rarely useful for marketing,
    so we constrain the search to a business-meaningful range.
    """
    window = metrics[(metrics["k"] >= k_min) & (metrics["k"] <= k_max)]
    return int(window.loc[window["silhouette"].idxmax(), "k"])


def fit_kmeans(
    rfm: pd.DataFrame, k: int, random_state: int = RANDOM_STATE
) -> tuple[ClusterModel, np.ndarray]:
    """Fit final K-means; return the model bundle and per-customer labels."""
    X, scaler = _fit_scaler_and_transform(rfm)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X)
    return ClusterModel(kmeans=km, scaler=scaler, k=k), km.labels_


def profile_clusters(rfm: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Mean RFM, size and revenue per cluster, sorted by mean Monetary desc."""
    df = rfm.copy()
    df["Cluster"] = labels
    profile = (
        df.groupby("Cluster")
        .agg(
            customers=("Recency", "size"),
            avg_recency=("Recency", "mean"),
            avg_frequency=("Frequency", "mean"),
            avg_monetary=("Monetary", "mean"),
            total_revenue=("Monetary", "sum"),
        )
        .reset_index()
        .sort_values("avg_monetary", ascending=False)
        .reset_index(drop=True)
    )
    profile["customer_share"] = profile["customers"] / profile["customers"].sum()
    profile["revenue_share"] = profile["total_revenue"] / profile["total_revenue"].sum()
    return profile


def name_clusters(profile: pd.DataFrame) -> dict[int, str]:
    """Heuristic, data-driven labels from each cluster's RFM position.

    Labels compare a cluster's mean R/F/M against the median across clusters,
    yielding readable names like 'High-Value Active' or 'Dormant Low-Value'.
    Unlike the rule grid these come straight from the data, so they describe
    *this* dataset's structure rather than a fixed marketing taxonomy.
    """
    med_r = profile["avg_recency"].median()
    med_f = profile["avg_frequency"].median()
    med_m = profile["avg_monetary"].median()

    names: dict[int, str] = {}
    for _, row in profile.iterrows():
        recent = row["avg_recency"] <= med_r  # lower recency = more active
        valuable = row["avg_monetary"] >= med_m
        frequent = row["avg_frequency"] >= med_f

        if valuable and recent:
            label = "High-Value Active"
        elif valuable and not recent:
            label = "High-Value Lapsing"
        elif recent and frequent:
            label = "Engaged Mid-Value"
        elif recent:
            label = "Recent Low-Value"
        else:
            label = "Dormant Low-Value"
        names[int(row["Cluster"])] = label
    return names

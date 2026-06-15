"""Unit tests for the K-means segmentation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from consumer_intel.segmentation import kmeans


@pytest.fixture
def rfm() -> pd.DataFrame:
    """Two clearly separated blobs so clustering is deterministic-ish."""
    rng = np.random.default_rng(1)
    low = pd.DataFrame(
        {
            "Recency": rng.integers(200, 365, 100),
            "Frequency": rng.integers(1, 3, 100),
            "Monetary": rng.uniform(5, 100, 100),
        }
    )
    high = pd.DataFrame(
        {
            "Recency": rng.integers(1, 30, 100),
            "Frequency": rng.integers(20, 50, 100),
            "Monetary": rng.uniform(2000, 5000, 100),
        }
    )
    return pd.concat([low, high], ignore_index=True)


def test_prepare_features_shape_and_scale(rfm):
    X = kmeans.prepare_features(rfm)
    assert X.shape == (len(rfm), 3)
    # standardised: each column ~ mean 0, std 1
    assert np.allclose(X.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(X.std(axis=0), 1, atol=1e-6)


def test_evaluate_k_range_returns_metrics(rfm):
    metrics = kmeans.evaluate_k_range(rfm, range(2, 5))
    assert list(metrics["k"]) == [2, 3, 4]
    assert metrics["inertia"].is_monotonic_decreasing  # inertia falls as k rises
    assert metrics["silhouette"].between(-1, 1).all()


def test_fit_kmeans_assigns_k_clusters(rfm):
    model, labels = kmeans.fit_kmeans(rfm, k=3)
    assert model.k == 3
    assert len(labels) == len(rfm)
    assert len(set(labels)) == 3


def test_profile_and_name_clusters(rfm):
    model, labels = kmeans.fit_kmeans(rfm, k=2)
    profile = kmeans.profile_clusters(rfm, labels)
    assert profile["customers"].sum() == len(rfm)
    # sorted by avg_monetary descending
    assert profile["avg_monetary"].is_monotonic_decreasing
    names = kmeans.name_clusters(profile)
    assert len(names) == 2
    # the top-monetary, recent blob should read as high value
    assert "High-Value" in names[int(profile.loc[0, "Cluster"])]


def test_choose_k_respects_window():
    metrics = pd.DataFrame(
        {
            "k": [2, 3, 4, 5, 6, 7, 8, 9, 10],
            "inertia": [100, 80, 70, 65, 62, 60, 59, 58, 57],
            "silhouette": [0.9, 0.4, 0.5, 0.45, 0.3, 0.2, 0.1, 0.05, 0.02],
        }
    )
    # best silhouette overall is k=2, but window is 3-8 -> should pick k=4
    assert kmeans.choose_k(metrics, k_min=3, k_max=8) == 4

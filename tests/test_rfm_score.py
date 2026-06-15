"""Unit tests for rule-based RFM scoring and the segment grid."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from consumer_intel.segmentation.rfm_score import (
    N_BINS,
    SEGMENT_ACTIONS,
    SEGMENT_GRID,
    assign_segment,
    score_rfm,
    summarize_segments,
)


@pytest.fixture
def rfm() -> pd.DataFrame:
    """200 synthetic customers spanning the RFM space for stable quantiles."""
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "Recency": rng.integers(1, 365, n),
            "Frequency": rng.integers(1, 50, n),
            "Monetary": rng.uniform(5, 5000, n),
        },
        index=[f"C{i:04d}" for i in range(n)],
    )


def test_grid_covers_all_25_cells():
    expected = {(r, fm) for r in range(1, 6) for fm in range(1, 6)}
    assert set(SEGMENT_GRID) == expected


def test_every_segment_has_an_action():
    for segment in set(SEGMENT_GRID.values()):
        assert segment in SEGMENT_ACTIONS


def test_assign_segment_corners():
    assert assign_segment(5, 5) == "Champions"
    assert assign_segment(1, 1) == "Lost"
    assert assign_segment(1, 5) == "Can't Lose Them"
    assert assign_segment(5, 1) == "New Customers"


def test_scores_are_in_range(rfm):
    scored = score_rfm(rfm)
    for col in ["R", "F", "M", "FM"]:
        assert scored[col].between(1, N_BINS).all()


def test_recency_score_is_reversed(rfm):
    # The most recent customer (smallest Recency) should get a high R score.
    scored = score_rfm(rfm)
    most_recent = scored["Recency"].idxmin()
    assert scored.loc[most_recent, "R"] >= 4


def test_score_rfm_adds_expected_columns(rfm):
    scored = score_rfm(rfm)
    for col in ["R", "F", "M", "FM", "RFM_Score", "Segment", "Action"]:
        assert col in scored.columns
    # Every customer maps to a known segment.
    assert scored["Segment"].isin(SEGMENT_ACTIONS).all()


def test_summarize_segments_shares_sum_to_one(rfm):
    summary = summarize_segments(score_rfm(rfm))
    assert summary["customer_share"].sum() == pytest.approx(1.0)
    assert summary["revenue_share"].sum() == pytest.approx(1.0)

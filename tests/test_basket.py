"""Unit tests for market basket analysis and Next Best Offer."""

from __future__ import annotations

import pandas as pd
import pytest

from consumer_intel.basket import nbo, rules


@pytest.fixture
def tx() -> pd.DataFrame:
    """Synthetic baskets with a planted association: A and B co-occur strongly.

    20 baskets:
      - 10 contain {A, B}            -> strong A<->B association
      - 5  contain {C, D}            -> separate C<->D association
      - 5  contain {A} only          -> dilutes A->B confidence a little
    Each line item needs a positive amount; values are irrelevant here.
    """
    rows = []
    inv = 0

    def add(items):
        nonlocal inv
        inv += 1
        for code in items:
            rows.append((str(inv), code, 1, 1.0))

    for _ in range(10):
        add(["A", "B"])
    for _ in range(5):
        add(["C", "D"])
    for _ in range(5):
        add(["A"])

    df = pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Quantity", "TotalPrice"])
    df["Description"] = df["StockCode"].map(
        {"A": "Apple", "B": "Banana", "C": "Carrot", "D": "Date"}
    )
    return df


@pytest.fixture
def rule_df(tx) -> pd.DataFrame:
    # low thresholds so the small fixture yields rules
    return rules.mine_rules(tx, min_item_support=0.1, min_support=0.1, min_threshold=1.0)


# --- rules -----------------------------------------------------------------
def test_frequent_products_prunes_rare(tx):
    keep = set(rules.frequent_products(tx, min_item_support=0.3))
    # A is in 15/20 baskets, B in 10/20 -> kept; C, D in 5/20 -> dropped at 0.3
    assert "A" in keep and "B" in keep
    assert "C" not in keep and "D" not in keep


def test_basket_matrix_is_invoice_by_product(tx):
    m = rules.build_basket_matrix(tx, min_item_support=0.1)
    assert set(m.columns) == {"A", "B", "C", "D"}
    # 20 baskets, all contain at least one frequent product
    assert m.shape[0] == 20


def test_mine_rules_finds_planted_association(rule_df):
    pairs = {
        (frozenset(r["antecedents"]), frozenset(r["consequents"])) for _, r in rule_df.iterrows()
    }
    assert (frozenset({"A"}), frozenset({"B"})) in pairs
    # A->B lift should exceed 1 (positive association)
    ab = rule_df[
        (rule_df["antecedents"] == frozenset({"A"})) & (rule_df["consequents"] == frozenset({"B"}))
    ]
    assert ab["lift"].iloc[0] > 1.0


def test_product_descriptions(tx):
    d = rules.product_descriptions(tx)
    assert d["A"] == "Apple"
    assert d["D"] == "Date"


# --- NBO -------------------------------------------------------------------
def test_recommend_for_product_returns_associated_item(rule_df):
    recs = nbo.recommend_for_product(rule_df, "A", top_n=5)
    assert "B" in set(recs["StockCode"])


def test_recommendation_excludes_items_in_basket(rule_df):
    recs = nbo.recommend_for_basket(rule_df, {"A", "B"}, top_n=5)
    # neither seed item should be recommended back
    assert "A" not in set(recs["StockCode"])
    assert "B" not in set(recs["StockCode"])


def test_recommend_unknown_product_is_empty(rule_df):
    recs = nbo.recommend_for_product(rule_df, "ZZZ_NOT_A_PRODUCT", top_n=5)
    assert recs.empty
    # still has the expected columns
    assert "StockCode" in recs.columns and "lift" in recs.columns


def test_add_descriptions(rule_df, tx):
    d = rules.product_descriptions(tx)
    recs = nbo.add_descriptions(nbo.recommend_for_product(rule_df, "A"), d)
    assert "Description" in recs.columns
    assert recs.loc[recs["StockCode"] == "B", "Description"].iloc[0] == "Banana"

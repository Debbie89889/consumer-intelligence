"""Next Best Offer (NBO) — turn association rules into recommendations.

Given a basket (or a single product), find the rules whose antecedents are
already satisfied by the basket, then rank the *new* consequent products by
lift. This is the simple, transparent cross-sell logic the brief asks for:
"given a basket or product, recommend the next product by lift".
"""

from __future__ import annotations

import pandas as pd


def _empty_recommendations() -> pd.DataFrame:
    return pd.DataFrame(columns=["StockCode", "lift", "confidence", "support", "from_rule"])


def recommend_for_basket(
    rules: pd.DataFrame,
    basket_items: set[str] | frozenset[str],
    top_n: int = 5,
) -> pd.DataFrame:
    """Recommend the next products for a basket, ranked by lift.

    Parameters
    ----------
    rules:
        Output of :func:`consumer_intel.basket.rules.mine_rules` (antecedents /
        consequents are frozensets of stock codes).
    basket_items:
        Stock codes currently in the basket.
    top_n:
        Number of recommendations to return.

    Returns
    -------
    DataFrame with one row per recommended ``StockCode`` and the best
    (max-lift) rule that produced it: ``lift``, ``confidence``, ``support`` and
    ``from_rule`` (the antecedent set that fired).
    """
    basket = frozenset(basket_items)
    best: dict[str, dict] = {}

    for _, rule in rules.iterrows():
        if not rule["antecedents"].issubset(basket):
            continue
        for item in rule["consequents"] - basket:
            cand = {
                "lift": float(rule["lift"]),
                "confidence": float(rule["confidence"]),
                "support": float(rule["support"]),
                "from_rule": ", ".join(sorted(rule["antecedents"])),
            }
            # keep the rule with the highest lift for each candidate item
            if item not in best or cand["lift"] > best[item]["lift"]:
                best[item] = cand

    if not best:
        return _empty_recommendations()

    out = (
        pd.DataFrame.from_dict(best, orient="index")
        .reset_index(names="StockCode")
        .sort_values("lift", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return out[["StockCode", "lift", "confidence", "support", "from_rule"]]


def recommend_for_product(rules: pd.DataFrame, product: str, top_n: int = 5) -> pd.DataFrame:
    """Recommend products to cross-sell with a single ``product``."""
    return recommend_for_basket(rules, {product}, top_n=top_n)


def add_descriptions(recommendations: pd.DataFrame, descriptions: dict[str, str]) -> pd.DataFrame:
    """Attach a human-readable ``Description`` column to recommendations."""
    out = recommendations.copy()
    out.insert(1, "Description", out["StockCode"].map(descriptions).fillna(""))
    return out

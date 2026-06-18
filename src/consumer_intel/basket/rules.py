"""Market basket analysis: frequent itemsets and association rules.

Pipeline: build a (sparse) invoice x product incidence matrix, mine frequent
itemsets with FP-Growth, then derive association rules scored by
support / confidence / lift.

Scale note: the cleaned data has ~36k invoices x ~4.6k products. A dense
one-hot matrix would be ~170M cells, so we (a) prune the long tail of rare
products before building the matrix and (b) keep it sparse. FP-Growth is used
over Apriori because it is far faster at this size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from scipy.sparse import csr_matrix

DEFAULT_MIN_ITEM_SUPPORT = 0.01  # keep products in >= 1% of baskets
DEFAULT_MIN_SUPPORT = 0.01  # min support for frequent itemsets
DEFAULT_MIN_LIFT = 1.0  # keep rules that beat independence


def frequent_products(
    transactions: pd.DataFrame, min_item_support: float = DEFAULT_MIN_ITEM_SUPPORT
) -> pd.Index:
    """Stock codes appearing in at least ``min_item_support`` of all baskets."""
    n_baskets = transactions["InvoiceNo"].nunique()
    by_product = transactions.groupby("StockCode")["InvoiceNo"].nunique()
    return by_product[by_product >= min_item_support * n_baskets].index


def build_basket_matrix(
    transactions: pd.DataFrame, min_item_support: float = DEFAULT_MIN_ITEM_SUPPORT
) -> pd.DataFrame:
    """Sparse one-hot invoice x product matrix over frequent products.

    Rows are invoices, columns are stock codes, values are booleans (was the
    product in that invoice). Rare products are pruned first to keep the matrix
    tractable; invoices left with no frequent products drop out naturally.
    """
    keep = frequent_products(transactions, min_item_support)
    sub = transactions[transactions["StockCode"].isin(keep)][
        ["InvoiceNo", "StockCode"]
    ].drop_duplicates()
    inv = sub["InvoiceNo"].astype("category")
    prod = sub["StockCode"].astype("category")
    mat = csr_matrix(
        (np.ones(len(sub), dtype=bool), (inv.cat.codes, prod.cat.codes)),
        shape=(len(inv.cat.categories), len(prod.cat.categories)),
    )
    return pd.DataFrame.sparse.from_spmatrix(mat, columns=prod.cat.categories)


def frequent_itemsets(
    basket: pd.DataFrame, min_support: float = DEFAULT_MIN_SUPPORT
) -> pd.DataFrame:
    """Frequent itemsets via FP-Growth (``itemsets`` as frozensets of codes)."""
    return fpgrowth(basket, min_support=min_support, use_colnames=True)


def association_rules_from(
    itemsets: pd.DataFrame, metric: str = "lift", min_threshold: float = DEFAULT_MIN_LIFT
) -> pd.DataFrame:
    """Association rules from itemsets, robust to the mlxtend API change.

    Newer mlxtend versions added a ``num_itemsets`` argument; we try the plain
    call first and fall back if the installed version requires it.
    """
    try:
        return association_rules(itemsets, metric=metric, min_threshold=min_threshold)
    except TypeError:
        return association_rules(
            itemsets, metric=metric, min_threshold=min_threshold, num_itemsets=len(itemsets)
        )


def mine_rules(
    transactions: pd.DataFrame,
    min_item_support: float = DEFAULT_MIN_ITEM_SUPPORT,
    min_support: float = DEFAULT_MIN_SUPPORT,
    metric: str = "lift",
    min_threshold: float = DEFAULT_MIN_LIFT,
) -> pd.DataFrame:
    """End-to-end: transactions -> association rules sorted by lift desc.

    Returned columns include ``antecedents`` / ``consequents`` (frozensets of
    stock codes), ``support``, ``confidence`` and ``lift``.
    """
    basket = build_basket_matrix(transactions, min_item_support=min_item_support)
    itemsets = frequent_itemsets(basket, min_support=min_support)
    rules = association_rules_from(itemsets, metric=metric, min_threshold=min_threshold)
    return rules.sort_values("lift", ascending=False).reset_index(drop=True)


def product_descriptions(transactions: pd.DataFrame) -> dict[str, str]:
    """Map each stock code to its most common description (for readable output)."""
    desc = (
        transactions.dropna(subset=["Description"])
        .groupby("StockCode")["Description"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
    )
    return desc.to_dict()

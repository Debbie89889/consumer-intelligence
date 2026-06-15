"""Shared test fixtures.

A tiny synthetic frame that exercises every cleaning rule, in the *raw* schema
(original column names) so tests cover ``standardize_columns`` too.
"""
# ruff: noqa: E501  -- tabular fixture rows are intentionally kept on one line each

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    """12 rows covering: valid sales, a duplicate, a cancellation, a return,
    an admin stock code, a zero price, and a missing customer id.

    Row guide:
      0,1 : identical valid UK sale (row 1 is an exact duplicate)
      2   : valid UK sale (5-digit + letter stock code)
      3   : valid EIRE sale
      4   : cancellation (invoice C..., negative qty)
      5   : return (negative qty, not a C invoice)
      6   : admin stock code POST
      7   : zero unit price
      8   : missing customer id
      9   : valid France sale
      10  : admin stock code M
      11  : valid France sale
    """
    rows = [
        (
            "489434",
            "85048",
            "GLASS BALL LIGHTS",
            12,
            "2009-12-01 07:45:00",
            6.95,
            "13085.0",
            "United Kingdom",
        ),
        (
            "489434",
            "85048",
            "GLASS BALL LIGHTS",
            12,
            "2009-12-01 07:45:00",
            6.95,
            "13085.0",
            "United Kingdom",
        ),
        (
            "489435",
            "79323P",
            "PINK CHERRY LIGHTS",
            5,
            "2009-12-01 08:00:00",
            6.75,
            "13085.0",
            "United Kingdom",
        ),
        ("489436", "22423", "REGENCY TEACUP", 3, "2009-12-02 10:00:00", 12.50, "14911.0", "EIRE"),
        ("C489437", "22423", "REGENCY TEACUP", -3, "2009-12-03 11:00:00", 12.50, "14911.0", "EIRE"),
        ("489438", "22423", "REGENCY TEACUP", -2, "2009-12-04 09:00:00", 12.50, "14911.0", "EIRE"),
        ("489439", "POST", "POSTAGE", 1, "2009-12-04 09:30:00", 18.00, "14911.0", "EIRE"),
        (
            "489440",
            "85048",
            "GLASS BALL LIGHTS",
            6,
            "2009-12-05 12:00:00",
            0.00,
            "13085.0",
            "United Kingdom",
        ),
        ("489441", "84997B", "CHILDRENS CUTLERY", 4, "2009-12-06 14:00:00", 4.15, None, "France"),
        (
            "489442",
            "84997B",
            "CHILDRENS CUTLERY",
            4,
            "2009-12-06 14:30:00",
            4.15,
            "12680.0",
            "France",
        ),
        ("489443", "M", "Manual", 1, "2009-12-07 10:00:00", 25.00, "12680.0", "France"),
        ("489444", "22423", "REGENCY TEACUP", 2, "2009-12-08 16:00:00", 12.50, "12680.0", "France"),
    ]
    cols = [
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
        "Country",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["Invoice"] = df["Invoice"].astype("string")
    df["StockCode"] = df["StockCode"].astype("string")
    df["Customer ID"] = df["Customer ID"].astype("string")
    df["Description"] = df["Description"].astype("string")
    df["Country"] = df["Country"].astype("string")
    return df

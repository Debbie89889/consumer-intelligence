"""Unit tests for the cleaning pipeline."""

from __future__ import annotations

import pandas as pd

from consumer_intel.data.clean import (
    add_total_price,
    clean_transactions,
    is_cancellation,
    is_product_stockcode,
    normalize_customer_id,
    standardize_columns,
)


def test_standardize_columns_renames_to_canonical(raw_sample):
    out = standardize_columns(raw_sample)
    assert "InvoiceNo" in out.columns
    assert "UnitPrice" in out.columns
    assert "CustomerID" in out.columns
    assert "Customer ID" not in out.columns


def test_is_cancellation_flags_c_prefix(raw_sample):
    df = standardize_columns(raw_sample)
    mask = is_cancellation(df)
    assert mask.sum() == 1
    assert df.loc[mask, "InvoiceNo"].iloc[0] == "C489437"


def test_is_product_stockcode_excludes_admin_codes(raw_sample):
    df = standardize_columns(raw_sample)
    mask = is_product_stockcode(df)
    kept_codes = set(df.loc[mask, "StockCode"])
    assert "POST" not in kept_codes
    assert "M" not in kept_codes
    assert "85048" in kept_codes
    assert "79323P" in kept_codes  # 5 digits + letter suffix is a product
    assert "84997B" in kept_codes


def test_normalize_customer_id_strips_decimal(raw_sample):
    df = standardize_columns(raw_sample)
    out = normalize_customer_id(df)
    assert "13085" in set(out["CustomerID"].dropna())
    assert out["CustomerID"].str.contains(r"\.", na=False).sum() == 0


def test_add_total_price():
    df = pd.DataFrame({"Quantity": [2, 3], "UnitPrice": [5.0, 4.0]})
    out = add_total_price(df)
    assert list(out["TotalPrice"]) == [10.0, 12.0]


def test_clean_transactions_keeps_only_valid_sales(raw_sample):
    result = clean_transactions(raw_sample)
    clean = result.transactions

    # Surviving rows: 2 distinct UK sales (the 3rd UK row was an exact dup),
    # 1 EIRE sale, and 2 France sales -> 5 rows.
    assert len(clean) == 5

    # No cancellations, no admin codes, all positive, all have a customer id.
    assert not is_cancellation(clean).any()
    assert is_product_stockcode(clean).all()
    assert (clean["Quantity"] > 0).all()
    assert (clean["UnitPrice"] > 0).all()
    assert clean["CustomerID"].notna().all()

    # TotalPrice present and correct.
    assert "TotalPrice" in clean.columns
    assert (clean["TotalPrice"] == clean["Quantity"] * clean["UnitPrice"]).all()


def test_clean_report_accounts_for_every_row(raw_sample):
    result = clean_transactions(raw_sample)
    report = result.report_frame()
    total_removed = report["rows_removed"].sum()
    assert result.n_raw - total_removed == result.n_clean
    # final 'rows_remaining' equals clean count
    assert report["rows_remaining"].iloc[-1] == result.n_clean


def test_dates_are_parsed(raw_sample):
    result = clean_transactions(raw_sample)
    assert pd.api.types.is_datetime64_any_dtype(result.transactions["InvoiceDate"])

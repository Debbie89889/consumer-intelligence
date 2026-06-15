"""Project-wide paths and constants.

Keeping these in one place means every module reads the same processed
parquet and applies the same cleaning conventions.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_CSV = RAW_DIR / "online_retail_II.csv"
CLEAN_PARQUET = PROCESSED_DIR / "transactions_clean.parquet"

# --- Schema ----------------------------------------------------------------
# The UCI/Kaggle CSV ships slightly different column names than the docs.
# We standardise once, here, and everything downstream uses the right side.
COLUMN_RENAME: dict[str, str] = {
    "Invoice": "InvoiceNo",
    "StockCode": "StockCode",
    "Description": "Description",
    "Quantity": "Quantity",
    "InvoiceDate": "InvoiceDate",
    "Price": "UnitPrice",
    "Customer ID": "CustomerID",
    "Country": "Country",
}

# --- Cleaning conventions --------------------------------------------------
# A cancelled order is flagged by an invoice number beginning with "C".
CANCELLATION_PREFIX = "C"

# Real product codes are a 5-digit number, optionally followed by 1-3 letters
# (e.g. "85048", "79323P", "84997B"). Everything else is an admin/non-product
# code: postage (POST/DOT), manual (M), bank charges, samples (S), discounts (D),
# Amazon fees, gift-card top-ups (gift_0001_*), test rows (TEST001), etc.
PRODUCT_STOCKCODE_PATTERN = r"^\d{5}[A-Za-z]{0,3}$"

"""Loading raw transactions and reading back the cleaned parquet.

Loaders only do I/O + dtype assignment. All business-rule cleaning lives in
``clean.py`` so the two concerns stay testable in isolation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from consumer_intel import config

# Read the raw text columns as strings so we never lose leading zeros on stock
# codes or let pandas guess a float for the (numeric-looking) customer id.
_RAW_DTYPES: dict[str, str] = {
    "Invoice": "string",
    "StockCode": "string",
    "Description": "string",
    "Quantity": "int64",
    "Price": "float64",
    "Customer ID": "string",
    "Country": "string",
}


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw Online Retail II CSV with stable dtypes.

    Parameters
    ----------
    path:
        Location of the raw CSV. Defaults to ``config.RAW_CSV``.

    Returns
    -------
    DataFrame with the original column names (``Invoice``, ``Price``,
    ``Customer ID`` ...) and ``InvoiceDate`` still a string. Dates are parsed
    during cleaning, not here.
    """
    csv_path = Path(path) if path is not None else config.RAW_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw CSV not found at {csv_path}. Download Online Retail II "
            "(UCI id=502 / Kaggle) into data/raw/ first."
        )
    return pd.read_csv(csv_path, dtype=_RAW_DTYPES)


def load_clean(path: str | Path | None = None) -> pd.DataFrame:
    """Read the cleaned transactions parquet produced by the Phase 0 pipeline."""
    parquet_path = Path(path) if path is not None else config.CLEAN_PARQUET
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Clean parquet not found at {parquet_path}. Run scripts/run_phase0.py to generate it."
        )
    return pd.read_parquet(parquet_path)

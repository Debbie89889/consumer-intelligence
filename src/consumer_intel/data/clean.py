"""Cleaning Online Retail II into an analysis-ready transactions table.

The public entry point is :func:`clean_transactions`. It returns a
:class:`CleanResult` carrying both the cleaned frame and a row-by-row report of
what each step removed, so the pipeline is auditable (and testable).

Design notes
------------
* Every helper is a pure function: ``DataFrame -> DataFrame`` (or
  ``-> Series[bool]``). No I/O, no global state.
* The default cleaning targets **customer-level positive sales** — the basis
  for RFM, CLV and basket analysis. Cancellations and returns are *removed*
  here, not analysed; later phases that need net revenue can re-derive them
  from the raw file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from consumer_intel import config


@dataclass
class CleanResult:
    """Cleaned transactions plus an audit trail of the cleaning run."""

    transactions: pd.DataFrame
    n_raw: int
    steps: list[tuple[str, int]] = field(default_factory=list)
    # each step is (description, rows_removed)

    @property
    def n_clean(self) -> int:
        return len(self.transactions)

    def report_frame(self) -> pd.DataFrame:
        """Tabular view of how many rows each step dropped."""
        rows = [("raw rows", 0, self.n_raw)]
        remaining = self.n_raw
        for desc, removed in self.steps:
            remaining -= removed
            rows.append((desc, removed, remaining))
        return pd.DataFrame(rows, columns=["step", "rows_removed", "rows_remaining"])


# --- Pure transform helpers ------------------------------------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename CSV columns to canonical names (see ``config.COLUMN_RENAME``)."""
    return df.rename(columns=config.COLUMN_RENAME)


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse ``InvoiceDate`` to datetime64 (no-op if already datetime)."""
    out = df.copy()
    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"])
    return out


def normalize_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a float-like id (``"13085.0"``) into a clean integer string.

    Missing ids become ``<NA>`` (pandas string NA), preserved for the later
    drop step rather than silently filled.
    """
    out = df.copy()
    numeric = pd.to_numeric(out["CustomerID"], errors="coerce").astype("Int64")
    out["CustomerID"] = numeric.astype("string").str.replace("<NA>", "", regex=False)
    out.loc[numeric.isna(), "CustomerID"] = pd.NA
    return out


def add_total_price(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``TotalPrice = Quantity * UnitPrice`` (line-item revenue)."""
    out = df.copy()
    out["TotalPrice"] = out["Quantity"] * out["UnitPrice"]
    return out


# --- Boolean masks (single responsibility, easy to unit test) --------------
def is_cancellation(df: pd.DataFrame) -> pd.Series:
    """True where the invoice is a cancellation (number starts with ``C``)."""
    return df["InvoiceNo"].str.startswith(config.CANCELLATION_PREFIX, na=False)


def is_product_stockcode(df: pd.DataFrame) -> pd.Series:
    """True where the stock code looks like a real product (not an admin code)."""
    return df["StockCode"].str.fullmatch(config.PRODUCT_STOCKCODE_PATTERN).fillna(False)


# --- Orchestrator ----------------------------------------------------------
def clean_transactions(df: pd.DataFrame) -> CleanResult:
    """Clean raw Online Retail II rows into an analysis-ready transactions table.

    Steps (each recorded in the returned report):

    1. standardise column names, parse dates, normalise customer id
    2. drop exact duplicate rows
    3. drop cancellations (invoice starting with ``C``)
    4. drop non-product / admin stock codes
    5. drop non-positive quantity (returns / adjustments)
    6. drop non-positive unit price (free items / data errors)
    7. drop rows with no customer id (can't attribute to a customer)

    Finally adds ``TotalPrice``.

    Parameters
    ----------
    df:
        Raw frame as returned by :func:`consumer_intel.data.load.load_raw`.

    Returns
    -------
    CleanResult
        ``.transactions`` is the cleaned frame; ``.report_frame()`` shows the
        rows removed at each step.
    """
    n_raw = len(df)
    steps: list[tuple[str, int]] = []

    df = standardize_columns(df)
    df = parse_dates(df)
    df = normalize_customer_id(df)

    def drop(mask_keep: pd.Series, label: str) -> pd.DataFrame:
        nonlocal df
        before = len(df)
        df = df.loc[mask_keep]
        steps.append((label, before - len(df)))
        return df

    # 2. exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    steps.append(("drop exact duplicates", before - len(df)))

    # 3-7. business-rule filters
    drop(~is_cancellation(df), "drop cancellations (invoice C*)")
    drop(is_product_stockcode(df), "drop non-product stock codes")
    drop(df["Quantity"] > 0, "drop non-positive quantity")
    drop(df["UnitPrice"] > 0, "drop non-positive unit price")
    drop(df["CustomerID"].notna(), "drop missing customer id")

    df = add_total_price(df).reset_index(drop=True)
    return CleanResult(transactions=df, n_raw=n_raw, steps=steps)

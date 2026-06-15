"""Phase 0 pipeline: load -> clean -> save parquet -> EDA -> report.

Run from the project root:

    python scripts/run_phase0.py

Outputs:
    data/processed/transactions_clean.parquet
    reports/eda_phase0.md
    reports/*.html   (interactive plotly charts)
"""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go

from consumer_intel import config
from consumer_intel.data.clean import clean_transactions
from consumer_intel.data.load import load_raw
from consumer_intel.eda import profile


def _fmt_money(x: float) -> str:
    return f"£{x:,.0f}"


def main() -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw CSV ...")
    raw = load_raw()
    print(f"  raw rows: {len(raw):,}")

    print("Cleaning ...")
    result = clean_transactions(raw)
    clean = result.transactions
    report = result.report_frame()
    print(report.to_string(index=False))
    print(f"  kept {result.n_clean:,} / {result.n_raw:,} ({result.n_clean / result.n_raw:.1%})")

    print(f"Writing parquet -> {config.CLEAN_PARQUET}")
    clean.to_parquet(config.CLEAN_PARQUET, index=False)

    # --- EDA tables --------------------------------------------------------
    overview = profile.dataset_overview(clean)
    monthly = profile.monthly_summary(clean)
    countries = profile.country_summary(clean, top_n=10)
    order_stats = profile.order_value_stats(clean)
    products = profile.top_products(clean, n=15)
    cust_dist = profile.customer_value_distribution(clean)

    # --- Charts ------------------------------------------------------------
    fig_month = go.Figure()
    fig_month.add_bar(x=monthly["month"], y=monthly["revenue"], name="Revenue")
    fig_month.add_scatter(
        x=monthly["month"],
        y=monthly["customers"],
        name="Active customers",
        yaxis="y2",
        mode="lines+markers",
    )
    fig_month.update_layout(
        title="Monthly revenue and active customers",
        yaxis=dict(title="Revenue (£)"),
        yaxis2=dict(title="Active customers", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    fig_month.write_html(config.REPORTS_DIR / "monthly_revenue.html")

    fig_ctry = px.bar(
        countries.iloc[::-1],
        x="revenue",
        y="Country",
        orientation="h",
        title="Top 10 countries by revenue",
    )
    fig_ctry.write_html(config.REPORTS_DIR / "country_revenue.html")

    # --- Markdown report ---------------------------------------------------
    lines: list[str] = []
    lines.append("# Phase 0 — EDA Report (Online Retail II)\n")
    lines.append("## Cleaning audit\n")
    lines.append(report.to_markdown(index=False))
    lines.append(
        f"\n**Kept {result.n_clean:,} of {result.n_raw:,} rows "
        f"({result.n_clean / result.n_raw:.1%}).**\n"
    )

    lines.append("## Dataset overview (cleaned)\n")
    lines.append(
        f"- Line items: **{overview['n_line_items']:,}**\n"
        f"- Customers: **{overview['n_customers']:,}**\n"
        f"- Invoices: **{overview['n_invoices']:,}**\n"
        f"- Products: **{overview['n_products']:,}**\n"
        f"- Countries: **{overview['n_countries']}**\n"
        f"- Date span: **{overview['date_min']:%Y-%m-%d} → {overview['date_max']:%Y-%m-%d}**\n"
        f"- Total revenue: **{_fmt_money(overview['total_revenue'])}**\n"
    )

    lines.append("## Top 10 countries by revenue\n")
    ctry_view = countries.copy()
    ctry_view["revenue"] = ctry_view["revenue"].map(_fmt_money)
    ctry_view["revenue_share"] = (ctry_view["revenue_share"] * 100).map(lambda v: f"{v:.1f}%")
    lines.append(ctry_view.to_markdown(index=False))

    lines.append("\n## Order value distribution (per invoice)\n")
    lines.append(order_stats.to_markdown(index=False))

    lines.append("\n## Per-customer spend distribution\n")
    lines.append(cust_dist.to_markdown(index=False))

    lines.append("\n## Top 15 products by revenue\n")
    prod_view = products.copy()
    prod_view["revenue"] = prod_view["revenue"].map(_fmt_money)
    lines.append(prod_view.to_markdown(index=False))

    lines.append("\n## Charts\n")
    lines.append("- `reports/monthly_revenue.html`\n- `reports/country_revenue.html`\n")

    (config.REPORTS_DIR / "eda_phase0.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report -> {config.REPORTS_DIR / 'eda_phase0.md'}")


if __name__ == "__main__":
    main()

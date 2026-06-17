"""Phase 2 pipeline: historical + predictive CLV, holdout validation, and
linkage of predicted CLV back to the Phase 1 segments.

Run from the project root (needs the Phase 0 clean parquet; Phase 1 parquet is
optional but used for the CLV-by-segment view):

    python scripts/run_phase2.py

Outputs:
    data/processed/customer_clv.parquet   (one row per customer)
    reports/phase2_clv.md
    reports/*.html   (interactive plotly charts)
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px

from consumer_intel import config
from consumer_intel.clv import historical, predictive, validate
from consumer_intel.data.load import load_clean

HORIZON_MONTHS = 3
HOLDOUT_DAYS = 180
SEGMENTS_PARQUET = config.PROCESSED_DIR / "customer_segments.parquet"
CLV_PARQUET = config.PROCESSED_DIR / "customer_clv.parquet"


def _fmt_money(x: float) -> str:
    return f"£{x:,.0f}"


def main() -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading clean transactions ...")
    tx = load_clean()
    obs_end = tx["InvoiceDate"].max()

    # --- historical (observed) CLV ----------------------------------------
    hist = historical.historical_clv(tx)
    print(f"  historical CLV for {len(hist):,} customers")

    # --- predictive CLV (BG/NBD + Gamma-Gamma) ----------------------------
    print(f"Fitting BG/NBD + Gamma-Gamma (horizon {HORIZON_MONTHS} months) ...")
    result = predictive.compute_predictive_clv(tx, horizon_months=HORIZON_MONTHS)
    preds = result.predictions
    corr = predictive.frequency_monetary_correlation(result.summary)
    n_elig = (preds["clv_method"] == "bg_nbd_gamma_gamma").sum()
    print(f"  eligible (repeat) customers: {n_elig:,} / {len(preds):,}")
    print(f"  frequency~monetary corr (Gamma-Gamma assumption): {corr:.4f}")

    # --- holdout validation -----------------------------------------------
    cal_end = obs_end - pd.Timedelta(days=HOLDOUT_DAYS)
    print(
        f"Validating on holdout: calibration <= {cal_end:%Y-%m-%d} < holdout <= {obs_end:%Y-%m-%d}"
    )
    val = validate.calibration_holdout(tx, calibration_end=cal_end, observation_end=obs_end)
    print(
        f"  MAE {val.metrics['mae']:.3f} | corr {val.metrics['correlation']:.3f} | "
        f"pred total {val.metrics['predicted_total_purchases']:.0f} vs "
        f"actual {val.metrics['actual_total_purchases']:.0f}"
    )

    # --- assemble per-customer table --------------------------------------
    customer = result.summary.join(preds).join(
        hist[["total_revenue", "n_orders", "avg_order_value", "tenure_days"]]
    )
    customer = customer.rename(columns={"total_revenue": "historical_clv"})

    # link Phase 1 segments if available
    seg_view = None
    if SEGMENTS_PARQUET.exists():
        seg = pd.read_parquet(SEGMENTS_PARQUET).set_index("CustomerID")
        customer = customer.join(seg[["Segment", "ClusterName"]])
        seg_view = (
            customer.groupby("Segment")
            .agg(
                customers=("predicted_clv", "size"),
                avg_predicted_clv=("predicted_clv", "mean"),
                total_predicted_clv=("predicted_clv", "sum"),
                avg_prob_alive=("prob_alive", "mean"),
            )
            .sort_values("total_predicted_clv", ascending=False)
            .reset_index()
        )

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    customer.reset_index().to_parquet(CLV_PARQUET, index=False)
    print(f"Wrote customer CLV table -> {CLV_PARQUET}")

    # --- charts -----------------------------------------------------------
    elig = customer[customer["clv_method"] == "bg_nbd_gamma_gamma"]
    fig_dist = px.histogram(
        elig[elig["predicted_clv"] > 0],
        x="predicted_clv",
        nbins=60,
        log_y=True,
        title=f"Predicted {HORIZON_MONTHS}-month CLV (repeat customers)",
    )
    fig_dist.write_html(config.REPORTS_DIR / "clv_distribution.html")

    fig_val = px.scatter(
        val.data,
        x="predicted_purchases",
        y="frequency_holdout",
        opacity=0.4,
        title="Holdout validation: predicted vs actual purchases",
        labels={"frequency_holdout": "actual holdout purchases"},
    )
    fig_val.write_html(config.REPORTS_DIR / "clv_validation.html")

    if seg_view is not None:
        fig_seg = px.bar(
            seg_view.sort_values("total_predicted_clv"),
            x="total_predicted_clv",
            y="Segment",
            orientation="h",
            title=f"Predicted {HORIZON_MONTHS}-month CLV by RFM segment",
        )
        fig_seg.write_html(config.REPORTS_DIR / "clv_by_segment.html")

    # --- markdown report --------------------------------------------------
    lines: list[str] = ["# Phase 2 — Customer Lifetime Value (BG/NBD + Gamma-Gamma)\n"]
    lines.append(
        f"Horizon **{HORIZON_MONTHS} months** · observation end **{obs_end:%Y-%m-%d}** · "
        f"**{len(customer):,}** customers "
        f"(**{n_elig:,}** repeat customers modelled, "
        f"{len(customer) - n_elig:,} one-time buyers on fallback)\n"
    )

    lines.append("## Model fit\n")
    lines.append("**BG/NBD (purchase frequency)**\n")
    lines.append(result.bgf.summary.round(4).to_markdown())
    lines.append("\n**Gamma-Gamma (monetary value)**\n")
    lines.append(result.ggf.summary.round(4).to_markdown())
    lines.append(
        f"\nGamma-Gamma assumption check — frequency~monetary correlation = "
        f"**{corr:.4f}** (close to 0 is good).\n"
    )

    lines.append("## Holdout validation (BG/NBD)\n")
    m = val.metrics
    lines.append(
        f"- Calibration ≤ {cal_end:%Y-%m-%d}, holdout {m['holdout_days']} days "
        f"to {obs_end:%Y-%m-%d}\n"
        f"- Customers: {m['n_customers']:,}\n"
        f"- MAE: **{m['mae']:.3f}** purchases · RMSE: {m['rmse']:.3f}\n"
        f"- Correlation (predicted vs actual): **{m['correlation']:.3f}**\n"
        f"- Aggregate: predicted **{m['predicted_total_purchases']:,.0f}** vs "
        f"actual **{m['actual_total_purchases']:,.0f}** purchases "
        f"({m['predicted_total_purchases'] / m['actual_total_purchases'] - 1:+.1%})\n"
    )

    lines.append("## Top 10 customers by predicted CLV\n")
    top = (
        customer.sort_values("predicted_clv", ascending=False)
        .head(10)[
            [
                "predicted_purchases",
                "prob_alive",
                "predicted_avg_value",
                "predicted_clv",
                "historical_clv",
                "clv_method",
            ]
        ]
        .copy()
    )
    top["predicted_purchases"] = top["predicted_purchases"].round(2)
    top["prob_alive"] = top["prob_alive"].round(3)
    for col in ["predicted_avg_value", "predicted_clv", "historical_clv"]:
        top[col] = top[col].map(_fmt_money)
    lines.append(top.reset_index().to_markdown(index=False))

    if seg_view is not None:
        lines.append("\n## Predicted CLV by RFM segment (Phase 1 linkage)\n")
        sv = seg_view.copy()
        sv["avg_predicted_clv"] = sv["avg_predicted_clv"].map(_fmt_money)
        sv["total_predicted_clv"] = sv["total_predicted_clv"].map(_fmt_money)
        sv["avg_prob_alive"] = sv["avg_prob_alive"].round(3)
        lines.append(sv.to_markdown(index=False))

    lines.append("\n## Charts\n")
    lines.append(
        "- `reports/clv_distribution.html`\n"
        "- `reports/clv_validation.html`\n"
        "- `reports/clv_by_segment.html`\n"
    )

    (config.REPORTS_DIR / "phase2_clv.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report -> {config.REPORTS_DIR / 'phase2_clv.md'}")


if __name__ == "__main__":
    main()

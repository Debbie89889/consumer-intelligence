"""Phase 1 pipeline: RFM -> rule-based segments + K-means clusters -> report.

Run from the project root (after run_phase0.py has produced the clean parquet):

    python scripts/run_phase1.py

Outputs:
    data/processed/customer_segments.parquet   (one row per customer)
    reports/phase1_segmentation.md
    reports/*.html   (interactive plotly charts)
"""

from __future__ import annotations

import plotly.express as px

from consumer_intel import config
from consumer_intel.data.load import load_clean
from consumer_intel.features.rfm import compute_rfm, snapshot_date
from consumer_intel.segmentation import kmeans, rfm_score

CUSTOMER_PARQUET = config.PROCESSED_DIR / "customer_segments.parquet"


def _fmt_money(x: float) -> str:
    return f"£{x:,.0f}"


def main() -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading clean transactions ...")
    tx = load_clean()
    snap = snapshot_date(tx)
    print(f"  snapshot date: {snap:%Y-%m-%d}")

    print("Computing RFM ...")
    rfm = compute_rfm(tx, snapshot=snap)
    print(f"  customers: {len(rfm):,}")

    # --- (a) rule-based RFM segments --------------------------------------
    scored = rfm_score.score_rfm(rfm)
    seg_summary = rfm_score.summarize_segments(scored)

    # --- (b) K-means -------------------------------------------------------
    print("Evaluating k = 2..10 ...")
    metrics = kmeans.evaluate_k_range(rfm, range(2, 11))
    best_k = kmeans.choose_k(metrics)
    print(metrics.to_string(index=False))
    print(f"  chosen k (best silhouette in 3-8): {best_k}")

    model, labels = kmeans.fit_kmeans(rfm, best_k)
    cluster_profile = kmeans.profile_clusters(rfm, labels)
    cluster_names = kmeans.name_clusters(cluster_profile)
    cluster_profile["ClusterName"] = cluster_profile["Cluster"].map(cluster_names)

    # --- assemble per-customer output -------------------------------------
    out = scored.copy()
    out["Cluster"] = labels
    out["ClusterName"] = out["Cluster"].map(cluster_names)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.reset_index().to_parquet(CUSTOMER_PARQUET, index=False)
    print(f"Wrote customer table -> {CUSTOMER_PARQUET}")

    # --- charts ------------------------------------------------------------
    fig_seg = px.bar(
        seg_summary.sort_values("total_revenue"),
        x="total_revenue",
        y="Segment",
        orientation="h",
        title="Revenue by RFM segment",
    )
    fig_seg.write_html(config.REPORTS_DIR / "rfm_segments_revenue.html")

    fig_elbow = px.line(metrics, x="k", y="inertia", markers=True, title="K-means elbow (inertia)")
    fig_elbow.write_html(config.REPORTS_DIR / "kmeans_elbow.html")

    fig_sil = px.line(metrics, x="k", y="silhouette", markers=True, title="K-means silhouette")
    fig_sil.write_html(config.REPORTS_DIR / "kmeans_silhouette.html")

    fig_scatter = px.scatter(
        out,
        x="Recency",
        y="Monetary",
        color="ClusterName",
        log_y=True,
        opacity=0.5,
        title="Customers by cluster (Recency vs Monetary)",
    )
    fig_scatter.write_html(config.REPORTS_DIR / "kmeans_scatter.html")

    # --- markdown report ---------------------------------------------------
    lines: list[str] = ["# Phase 1 — Customer Segmentation (RFM + K-means)\n"]
    lines.append(
        f"Snapshot date **{snap:%Y-%m-%d}** · **{len(rfm):,}** customers · "
        f"total revenue **{_fmt_money(rfm['Monetary'].sum())}**\n"
    )

    lines.append("## (a) Rule-based RFM segments\n")
    sv = seg_summary.copy()
    sv["avg_recency"] = sv["avg_recency"].round(0).astype(int)
    sv["avg_frequency"] = sv["avg_frequency"].round(1)
    sv["avg_monetary"] = sv["avg_monetary"].map(_fmt_money)
    sv["total_revenue"] = sv["total_revenue"].map(_fmt_money)
    sv["customer_share"] = (sv["customer_share"] * 100).map(lambda v: f"{v:.1f}%")
    sv["revenue_share"] = (sv["revenue_share"] * 100).map(lambda v: f"{v:.1f}%")
    lines.append(sv.to_markdown(index=False))

    lines.append("\n### Recommended action per segment\n")
    actions = scored[["Segment", "Action"]].drop_duplicates().sort_values("Segment")
    lines.append(actions.to_markdown(index=False))

    lines.append("\n## (b) K-means clusters\n")
    lines.append(
        f"Chose **k = {best_k}** (highest silhouette in the 3-8 range). "
        "Features: `log1p`-transformed, standardised Recency/Frequency/Monetary.\n"
    )
    cp = cluster_profile.copy()
    cp["avg_recency"] = cp["avg_recency"].round(0).astype(int)
    cp["avg_frequency"] = cp["avg_frequency"].round(1)
    cp["avg_monetary"] = cp["avg_monetary"].map(_fmt_money)
    cp["total_revenue"] = cp["total_revenue"].map(_fmt_money)
    cp["customer_share"] = (cp["customer_share"] * 100).map(lambda v: f"{v:.1f}%")
    cp["revenue_share"] = (cp["revenue_share"] * 100).map(lambda v: f"{v:.1f}%")
    cp = cp[
        [
            "Cluster",
            "ClusterName",
            "customers",
            "customer_share",
            "avg_recency",
            "avg_frequency",
            "avg_monetary",
            "total_revenue",
            "revenue_share",
        ]
    ]
    lines.append(cp.to_markdown(index=False))

    lines.append("\n## k-selection metrics\n")
    mv = metrics.copy()
    mv["inertia"] = mv["inertia"].round(0).astype(int)
    mv["silhouette"] = mv["silhouette"].round(3)
    lines.append(mv.to_markdown(index=False))

    lines.append("\n## Charts\n")
    lines.append(
        "- `reports/rfm_segments_revenue.html`\n"
        "- `reports/kmeans_elbow.html`\n"
        "- `reports/kmeans_silhouette.html`\n"
        "- `reports/kmeans_scatter.html`\n"
    )

    (config.REPORTS_DIR / "phase1_segmentation.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report -> {config.REPORTS_DIR / 'phase1_segmentation.md'}")


if __name__ == "__main__":
    main()

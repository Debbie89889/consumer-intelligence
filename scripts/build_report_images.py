"""Render static PNG charts for the Markdown reports from the committed parquets.

GitHub renders PNGs inline (unlike the large interactive .html), so the reports
stay viewable straight from the repo. Charts use English labels to avoid font
issues; the Markdown around them provides the Chinese narrative.

    python scripts/build_report_images.py        # writes reports/*.png

Charts that need holdout actuals or a trained model (CLV holdout validation,
SHAP) are not rebuilt here; their figures from the numbers remain in the text.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from consumer_intel import config

REPORTS = config.REPORTS_DIR
TEAL, NAVY, AMBER = "#2a9d8f", "#0f2a4a", "#e9a23b"
W, H, SCALE = 900, 470, 2


def save(fig, name: str, *, h: int = H) -> None:
    fig.update_layout(
        template="plotly_white",
        font=dict(size=13),
        title_font=dict(size=17, color=NAVY),
        margin=dict(l=60, r=30, t=50, b=50),
    )
    path = REPORTS / f"{name}.png"
    fig.write_image(str(path), width=W, height=h, scale=SCALE)
    print(f"  {name}.png")


def p(name: str) -> pd.DataFrame:
    return pd.read_parquet(config.PROCESSED_DIR / f"{name}.parquet")


def phase0() -> None:
    m = p("monthly_summary")
    fig = px.area(
        m,
        x="month",
        y="revenue",
        title="Monthly Revenue",
        labels={"month": "Month", "revenue": "Revenue (£)"},
        color_discrete_sequence=[TEAL],
    )
    fig.update_traces(line=dict(width=2), fillcolor="rgba(42,157,143,.15)")
    save(fig, "monthly_revenue")

    c = p("country_summary").sort_values("revenue", ascending=False).head(15)
    fig = px.bar(
        c.sort_values("revenue"),
        x="revenue",
        y="country",
        orientation="h",
        title="Revenue by Country (Top 15)",
        labels={"revenue": "Revenue (£)", "country": ""},
        color="revenue",
        color_continuous_scale=["#7cc6b8", NAVY],
    )
    fig.update_layout(coloraxis_showscale=False)
    save(fig, "country_revenue", h=520)


def phase1() -> None:
    cs = p("customer_segments")

    rev = cs.groupby("Segment")["Monetary"].sum().sort_values()
    fig = px.bar(
        x=rev.values,
        y=rev.index,
        orientation="h",
        title="Revenue by RFM Segment",
        labels={"x": "Revenue (£)", "y": ""},
        color=rev.values,
        color_continuous_scale=["#7cc6b8", NAVY],
    )
    fig.update_layout(coloraxis_showscale=False)
    save(fig, "rfm_segments_revenue", h=520)

    fig = px.scatter(
        cs,
        x="Recency",
        y="Monetary",
        color="ClusterName",
        title="Customer Clusters (Recency vs Monetary)",
        labels={"Recency": "Recency (days)", "Monetary": "Monetary (£, log)"},
        opacity=0.55,
        log_y=True,
    )
    save(fig, "kmeans_scatter")

    x = StandardScaler().fit_transform(np.log1p(cs[["Recency", "Frequency", "Monetary"]]))
    ks = list(range(2, 9))
    inertia, sil = [], []
    from sklearn.metrics import silhouette_score

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(x)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(x, km.labels_))
    fig = px.line(
        x=ks,
        y=inertia,
        markers=True,
        title="K-means Elbow (inertia)",
        labels={"x": "k (clusters)", "y": "Inertia"},
        color_discrete_sequence=[NAVY],
    )
    fig.add_vline(
        x=4,
        line_dash="dot",
        line_color=AMBER,
        annotation_text="chosen k=4",
        annotation_position="top",
    )
    save(fig, "kmeans_elbow", h=380)
    fig = px.line(
        x=ks,
        y=sil,
        markers=True,
        title="K-means Silhouette",
        labels={"x": "k (clusters)", "y": "Silhouette score"},
        color_discrete_sequence=[TEAL],
    )
    fig.add_vline(
        x=4,
        line_dash="dot",
        line_color=AMBER,
        annotation_text="chosen k=4",
        annotation_position="top",
    )
    save(fig, "kmeans_silhouette", h=380)


def phase2() -> None:
    cv = p("customer_clv")
    cap = cv["predicted_clv"].quantile(0.99)
    fig = px.histogram(
        cv[cv["predicted_clv"] <= cap],
        x="predicted_clv",
        nbins=60,
        title="Predicted CLV Distribution (≤ 99th pct)",
        labels={"predicted_clv": "Predicted CLV (£)"},
        color_discrete_sequence=[TEAL],
    )
    save(fig, "clv_distribution")

    order = cv.groupby("Segment")["predicted_clv"].median().sort_values().index
    fig = px.box(
        cv,
        x="Segment",
        y="predicted_clv",
        title="Predicted CLV by Segment",
        labels={"Segment": "", "predicted_clv": "Predicted CLV (£, log)"},
        category_orders={"Segment": list(order)},
        log_y=True,
        color_discrete_sequence=[NAVY],
    )
    top = float(cv["predicted_clv"].quantile(0.999))
    fig.update_yaxes(range=[-0.3, float(np.log10(top)) + 0.2])
    fig.update_xaxes(tickangle=-30)
    save(fig, "clv_by_segment", h=500)


def phase3() -> None:
    ar = p("association_rules")
    fig = px.scatter(
        ar,
        x="support",
        y="confidence",
        color="lift",
        size="lift",
        title="Association Rules (support × confidence, colored by lift)",
        labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"},
        color_continuous_scale="Plasma",
        opacity=0.7,
        size_max=18,
    )
    save(fig, "basket_rules_scatter")


def phase4() -> None:
    # Phase 4 performance charts (ROC/calibration) are intentionally NOT rendered
    # here: propensity_scores.parquet holds in-sample scores for the full modelling
    # population, so any metric computed on it is optimistic (≈0.94 AUC) and would
    # contradict the honest held-out test AUC (0.804) reported in the text. Faithful
    # reproduction needs the original train/test split, which is not shipped.
    return


def main() -> None:
    print("Rendering report images ->", REPORTS)
    phase0()
    phase1()
    phase2()
    phase3()
    phase4()
    print("done.")


if __name__ == "__main__":
    main()

"""Phase 4 pipeline: build the propensity training table, fit a logistic
baseline and a LightGBM model, evaluate (ROC-AUC, PR-AUC, calibration) and
explain the tree model with SHAP.

Run from the project root (needs the Phase 0 clean parquet):

    python scripts/run_phase4.py

Outputs:
    data/processed/propensity_scores.parquet   (per-customer score + label)
    reports/phase4_propensity.md
    reports/*.html
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from consumer_intel import config
from consumer_intel.data.load import load_clean
from consumer_intel.propensity import explain, model
from consumer_intel.propensity.features import (
    LABEL_COLUMN,
    build_training_table,
    default_cutoff,
)

HORIZON_DAYS = 90
SCORES_PARQUET = config.PROCESSED_DIR / "propensity_scores.parquet"


def main() -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading clean transactions ...")
    tx = load_clean()
    cutoff = default_cutoff(tx, HORIZON_DAYS)
    print(f"  cutoff {cutoff:%Y-%m-%d}, horizon {HORIZON_DAYS}d")

    table = build_training_table(tx, cutoff, horizon_days=HORIZON_DAYS)
    X, y = model.split_xy(table)
    print(f"  {len(table):,} customers, positive rate {y.mean():.3f}")

    X_train, X_test, y_train, y_test = model.train_test(X, y)

    print("Fitting logistic baseline ...")
    logit = model.fit_logistic(X_train, y_train)
    logit_metrics = model.evaluate(logit, X_test, y_test)

    print("Fitting LightGBM ...")
    lgbm = model.fit_lightgbm(X_train, y_train)
    lgbm_metrics = model.evaluate(lgbm, X_test, y_test)

    print(
        f"  LogReg   ROC-AUC {logit_metrics['roc_auc']:.3f} | PR-AUC {logit_metrics['pr_auc']:.3f}"
    )
    print(f"  LightGBM ROC-AUC {lgbm_metrics['roc_auc']:.3f} | PR-AUC {lgbm_metrics['pr_auc']:.3f}")

    calib = model.calibration_table(lgbm, X_test, y_test, n_bins=10)
    shap_imp = explain.shap_importance(lgbm, X_test)

    # --- score the full population (for downstream targeting) --------------
    scores = table.copy()
    scores["propensity"] = model.predict_proba(lgbm, X)
    scores[["propensity", LABEL_COLUMN]].reset_index().to_parquet(SCORES_PARQUET, index=False)
    print(f"Wrote scores -> {SCORES_PARQUET}")

    # --- charts ------------------------------------------------------------
    fig_cal = go.Figure()
    fig_cal.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect", line=dict(dash="dash"))
    fig_cal.add_scatter(
        x=calib["mean_predicted"],
        y=calib["fraction_positive"],
        mode="lines+markers",
        name="LightGBM",
    )
    fig_cal.update_layout(
        title="Calibration (LightGBM)",
        xaxis_title="Mean predicted probability",
        yaxis_title="Observed fraction positive",
    )
    fig_cal.write_html(config.REPORTS_DIR / "propensity_calibration.html")

    fig_shap = px.bar(
        shap_imp.iloc[::-1],
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title="SHAP feature importance (LightGBM)",
    )
    fig_shap.write_html(config.REPORTS_DIR / "propensity_shap.html")

    # --- report ------------------------------------------------------------
    lines: list[str] = ["# Phase 4 — Purchase Propensity (will they buy in 90 days?)\n"]
    lines.append(
        f"Cutoff **{cutoff:%Y-%m-%d}**, horizon **{HORIZON_DAYS} days**. "
        f"**{len(table):,}** customers active before the cutoff; "
        f"positive rate **{y.mean():.1%}**. Features use only pre-cutoff data "
        f"(no leakage); label is a purchase in the {HORIZON_DAYS}-day window after.\n"
    )

    lines.append("## Model comparison (held-out test set)\n")
    cmp = pd.DataFrame(
        [
            {"model": "Logistic (baseline)", **logit_metrics},
            {"model": "LightGBM", **lgbm_metrics},
        ]
    )[["model", "roc_auc", "pr_auc", "brier", "base_rate", "n_test"]]
    cmp[["roc_auc", "pr_auc", "brier", "base_rate"]] = cmp[
        ["roc_auc", "pr_auc", "brier", "base_rate"]
    ].round(3)
    lines.append(cmp.to_markdown(index=False))
    lines.append(
        "\nHigher ROC-AUC/PR-AUC is better; lower Brier (calibration error) is "
        "better. PR-AUC is the more honest headline here since we care about "
        "ranking likely buyers.\n"
    )

    lines.append("## SHAP feature importance (LightGBM)\n")
    si = shap_imp.copy()
    si["mean_abs_shap"] = si["mean_abs_shap"].round(4)
    lines.append(si.to_markdown(index=False))

    lines.append("\n## Calibration (LightGBM)\n")
    cv = calib.round(3)
    lines.append(cv.to_markdown(index=False))

    lines.append("\n## Charts\n")
    lines.append("- `reports/propensity_calibration.html`\n- `reports/propensity_shap.html`\n")

    (config.REPORTS_DIR / "phase4_propensity.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report -> {config.REPORTS_DIR / 'phase4_propensity.md'}")


if __name__ == "__main__":
    main()

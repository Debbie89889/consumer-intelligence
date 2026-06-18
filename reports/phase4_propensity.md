# Phase 4 — Purchase Propensity (will they buy in 90 days?)

Cutoff **2011-09-10**, horizon **90 days**. **5,256** customers active before the cutoff; positive rate **43.6%**. Features use only pre-cutoff data (no leakage); label is a purchase in the 90-day window after.

## Model comparison (held-out test set)

| model               |   roc_auc |   pr_auc |   brier |   base_rate |   n_test |
|:--------------------|----------:|---------:|--------:|------------:|---------:|
| Logistic (baseline) |     0.804 |    0.781 |   0.178 |       0.435 |     1314 |
| LightGBM            |     0.784 |    0.759 |   0.188 |       0.435 |     1314 |

Higher ROC-AUC/PR-AUC is better; lower Brier (calibration error) is better. PR-AUC is the more honest headline here since we care about ranking likely buyers.

## SHAP feature importance (LightGBM)

| feature                |   mean_abs_shap |
|:-----------------------|----------------:|
| recency_days           |          0.6585 |
| recency_over_tenure    |          0.468  |
| avg_interpurchase_days |          0.3379 |
| monetary               |          0.2905 |
| avg_order_value        |          0.1964 |
| tenure_days            |          0.171  |
| avg_basket_size        |          0.1684 |
| frequency              |          0.1655 |
| distinct_products      |          0.1404 |

## Calibration (LightGBM)

|   mean_predicted |   fraction_positive |
|-----------------:|--------------------:|
|            0.035 |               0.106 |
|            0.103 |               0.183 |
|            0.166 |               0.214 |
|            0.234 |               0.326 |
|            0.307 |               0.351 |
|            0.42  |               0.389 |
|            0.535 |               0.492 |
|            0.683 |               0.626 |
|            0.832 |               0.733 |
|            0.965 |               0.932 |

## Charts

- `reports/propensity_calibration.html`
- `reports/propensity_shap.html`

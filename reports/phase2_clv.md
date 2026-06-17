# Phase 2 — Customer Lifetime Value (BG/NBD + Gamma-Gamma)

Horizon **3 months** · observation end **2011-12-09** · **5,852** customers (**4,179** repeat customers modelled, 1,673 one-time buyers on fallback)

## Model fit

**BG/NBD (purchase frequency)**

|       |    coef |   se(coef) |   lower 95% bound |   upper 95% bound |
|:------|--------:|-----------:|------------------:|------------------:|
| r     |  0.6708 |     0.016  |            0.6394 |            0.7023 |
| alpha | 63.7407 |     1.9546 |           59.9097 |           67.5717 |
| a     |  0.0742 |     0.0076 |            0.0594 |            0.089  |
| b     |  1.2294 |     0.1509 |            0.9336 |            1.5253 |

**Gamma-Gamma (monetary value)**

|    |    coef |   se(coef) |   lower 95% bound |   upper 95% bound |
|:---|--------:|-----------:|------------------:|------------------:|
| p  | 11.7164 |     0.2251 |           11.2753 |           12.1576 |
| q  |  0.8936 |     0.0172 |            0.8599 |            0.9273 |
| v  | 11.6947 |     0.2302 |           11.2434 |           12.146  |

Gamma-Gamma assumption check — frequency~monetary correlation = **0.0203** (close to 0 is good).

## Holdout validation (BG/NBD)

- Calibration ≤ 2011-06-12, holdout 180 days to 2011-12-09
- Customers: 4,952
- MAE: **1.039** purchases · RMSE: 1.783
- Correlation (predicted vs actual): **0.848**
- Aggregate: predicted **7,549** vs actual **7,717** purchases (-2.2%)

## Top 10 customers by predicted CLV

|   CustomerID |   predicted_purchases |   prob_alive | predicted_avg_value   | predicted_clv   | historical_clv   | clv_method         |
|-------------:|----------------------:|-------------:|:----------------------|:----------------|:-----------------|:-------------------|
|        16446 |                  0.53 |        0.943 | £170,026              | £86,840         | £168,472         | bg_nbd_gamma_gamma |
|        18102 |                  7.55 |        0.999 | £8,770                | £64,012         | £580,987         | bg_nbd_gamma_gamma |
|        14646 |                 10.29 |        0.999 | £5,790                | £57,561         | £526,752         | bg_nbd_gamma_gamma |
|        17450 |                  5.52 |        0.996 | £6,854                | £36,586         | £244,784         | bg_nbd_gamma_gamma |
|        14156 |                 13.2  |        0.998 | £2,604                | £33,218         | £303,070         | bg_nbd_gamma_gamma |
|        14911 |                 27.97 |        1     | £1,104                | £29,832         | £272,253         | bg_nbd_gamma_gamma |
|        14096 |                  9.01 |        0.993 | £3,331                | £29,020         | £53,258          | bg_nbd_gamma_gamma |
|        12415 |                  2.84 |        0.991 | £7,877                | £21,656         | £144,033         | bg_nbd_gamma_gamma |
|        13694 |                  9.4  |        0.999 | £2,355                | £21,397         | £195,641         | bg_nbd_gamma_gamma |
|        17511 |                  5.74 |        0.998 | £3,381                | £18,771         | £172,133         | bg_nbd_gamma_gamma |

## Predicted CLV by RFM segment (Phase 1 linkage)

| Segment            |   customers | avg_predicted_clv   | total_predicted_clv   |   avg_prob_alive |
|:-------------------|------------:|:--------------------|:----------------------|-----------------:|
| Champions          |        1114 | £1,346              | £1,499,877            |            0.988 |
| Loyal Customers    |        1143 | £346                | £395,529              |            0.958 |
| Potential Loyalist |         565 | £218                | £123,355              |            0.958 |
| At Risk            |         413 | £186                | £76,764               |            0.669 |
| Promising          |         442 | £133                | £58,979               |            0.952 |
| About to Sleep     |         713 | £63                 | £45,070               |            0.9   |
| Need Attention     |         293 | £121                | £35,577               |            0.841 |
| New Customers      |          95 | £296                | £28,159               |            1     |
| Hibernating        |         629 | £39                 | £24,654               |            0.852 |
| Lost               |         315 | £44                 | £14,010               |            1     |
| Can't Lose Them    |         130 | £74                 | £9,667                |            0.38  |

## Charts

- `reports/clv_distribution.html`
- `reports/clv_validation.html`
- `reports/clv_by_segment.html`

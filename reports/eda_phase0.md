# Phase 0 — EDA Report (Online Retail II)

## Cleaning audit

| step                            |   rows_removed |   rows_remaining |
|:--------------------------------|---------------:|-----------------:|
| raw rows                        |              0 |          1067371 |
| drop exact duplicates           |          34335 |          1033036 |
| drop cancellations (invoice C*) |          19104 |          1013932 |
| drop non-product stock codes    |           4791 |          1009141 |
| drop non-positive quantity      |           3362 |          1005779 |
| drop non-positive unit price    |           2566 |          1003213 |
| drop missing customer id        |         226636 |           776577 |

**Kept 776,577 of 1,067,371 rows (72.8%).**

## Dataset overview (cleaned)

- Line items: **776,577**
- Customers: **5,852**
- Invoices: **36,594**
- Products: **4,619**
- Countries: **41**
- Date span: **2009-12-01 → 2011-12-09**
- Total revenue: **£17,068,568**

## Top 10 countries by revenue

| Country        | revenue     |   orders |   customers | revenue_share   |
|:---------------|:------------|---------:|------------:|:----------------|
| United Kingdom | £14,288,759 |    33361 |        5334 | 83.7%           |
| EIRE           | £586,626    |      528 |           3 | 3.4%            |
| Netherlands    | £549,773    |      216 |          22 | 3.2%            |
| Germany        | £383,289    |      753 |         107 | 2.2%            |
| France         | £309,403    |      590 |          93 | 1.8%            |
| Australia      | £167,800    |       89 |          15 | 1.0%            |
| Spain          | £97,767     |      144 |          38 | 0.6%            |
| Switzerland    | £93,401     |       82 |          22 | 0.5%            |
| Sweden         | £86,045     |       98 |          19 | 0.5%            |
| Denmark        | £67,423     |       42 |          12 | 0.4%            |

## Order value distribution (per invoice)

| stat   |   order_value |
|:-------|--------------:|
| count  |     36594     |
| mean   |       466.431 |
| std    |      1358.87  |
| min    |         0.38  |
| 25%    |       158.63  |
| 50%    |       302.55  |
| 75%    |       473.88  |
| 90%    |       824.553 |
| 99%    |      3528.02  |
| max    |    168470     |

## Per-customer spend distribution

| stat   |   customer_spend |
|:-------|-----------------:|
| count  |         5852     |
| mean   |         2916.71  |
| std    |        14306.9   |
| min    |            2.95  |
| 25%    |          339.575 |
| 50%    |          856.02  |
| 75%    |         2241.03  |
| 90%    |         5392.77  |
| 99%    |        28631.1   |
| max    |       580987     |

## Top 15 products by revenue

| StockCode   | revenue   |   units | description                        |
|:------------|:----------|--------:|:-----------------------------------|
| 22423       | £277,656  |   24124 | REGENCY CAKESTAND 3 TIER           |
| 85123A      | £247,203  |   91814 | WHITE HANGING HEART T-LIGHT HOLDER |
| 23843       | £168,470  |   80995 | PAPER CRAFT , LITTLE BIRDIE        |
| 85099B      | £167,921  |   93436 | JUMBO BAG RED RETROSPOT            |
| 84879       | £124,352  |   78234 | ASSORTED COLOUR BIRD ORNAMENT      |
| 47566       | £103,283  |   23460 | PARTY BUNTING                      |
| 23166       | £81,417   |   77916 | MEDIUM CERAMIC TOP STORAGE JAR     |
| 22086       | £76,598   |   28380 | PAPER CHAIN KIT 50'S CHRISTMAS     |
| 79321       | £69,084   |   14843 | CHILLI LIGHTS                      |
| 22386       | £67,770   |   37338 | JUMBO BAG PINK POLKADOT            |
| 48138       | £64,270   |    9879 | DOORMAT UNION FLAG                 |
| 85099F      | £64,128   |   35842 | JUMBO BAG STRAWBERRY               |
| 21137       | £63,092   |   18417 | BLACK RECORD COVER FRAME           |
| 20685       | £60,547   |    9364 | DOORMAT RED RETROSPOT              |
| 20725       | £59,375   |   37750 | LUNCH BAG RED RETROSPOT            |

## Charts

- `reports/monthly_revenue.html`
- `reports/country_revenue.html`

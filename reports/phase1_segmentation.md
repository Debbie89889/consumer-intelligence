# Phase 1 — Customer Segmentation (RFM + K-means)

Snapshot date **2011-12-10** · **5,852** customers · total revenue **£17,068,568**

## (a) Rule-based RFM segments

| Segment            |   customers |   avg_recency |   avg_frequency | avg_monetary   | total_revenue   | customer_share   | revenue_share   |
|:-------------------|------------:|--------------:|----------------:|:---------------|:----------------|:-----------------|:----------------|
| Champions          |        1114 |            14 |            18.1 | £10,030        | £11,173,636     | 19.0%            | 65.5%           |
| Loyal Customers    |        1143 |            70 |             6.9 | £2,639         | £3,016,107      | 19.5%            | 17.7%           |
| At Risk            |         413 |           342 |             6   | £2,737         | £1,130,572      | 7.1%             | 6.6%            |
| Can't Lose Them    |         130 |           491 |             6   | £2,916         | £379,138        | 2.2%             | 2.2%            |
| Potential Loyalist |         565 |            24 |             2.2 | £550           | £310,944        | 9.7%             | 1.8%            |
| Need Attention     |         293 |           204 |             3.2 | £976           | £286,079        | 5.0%             | 1.7%            |
| About to Sleep     |         713 |           314 |             1.5 | £376           | £268,354        | 12.2%            | 1.6%            |
| Hibernating        |         629 |           546 |             1.4 | £413           | £259,872        | 10.7%            | 1.5%            |
| Promising          |         442 |           108 |             1.6 | £411           | £181,856        | 7.6%             | 1.1%            |
| Lost               |         315 |           567 |             1   | £147           | £46,417         | 5.4%             | 0.3%            |
| New Customers      |          95 |            29 |             1   | £164           | £15,593         | 1.6%             | 0.1%            |

### Recommended action per segment

| Segment            | Action                                                    |
|:-------------------|:----------------------------------------------------------|
| About to Sleep     | Reactivation nudges before they churn.                    |
| At Risk            | Win-back: personalised offers, remind them of value.      |
| Can't Lose Them    | High-touch win-back; they were valuable, don't lose them. |
| Champions          | Reward loyalty; early access, referrals, VIP perks.       |
| Hibernating        | Low-cost reactivation; otherwise deprioritise spend.      |
| Lost               | Minimal spend; only broad, cheap campaigns.               |
| Loyal Customers    | Upsell higher-value lines; ask for reviews.               |
| Need Attention     | Time-limited offers on recently browsed / bought lines.   |
| New Customers      | Strong onboarding; make the second purchase easy.         |
| Potential Loyalist | Membership / loyalty programme to deepen the habit.       |
| Promising          | Nurture with targeted offers to build frequency.          |

## (b) K-means clusters

Chose **k = 4** (highest silhouette in the 3-8 range). Features: `log1p`-transformed, standardised Recency/Frequency/Monetary.

|   Cluster | ClusterName        |   customers | customer_share   |   avg_recency |   avg_frequency | avg_monetary   | total_revenue   | revenue_share   |
|----------:|:-------------------|------------:|:-----------------|--------------:|----------------:|:---------------|:----------------|:----------------|
|         1 | High-Value Active  |        1182 | 20.2%            |            28 |            19.2 | £10,590        | £12,517,347     | 73.3%           |
|         0 | High-Value Lapsing |        1455 | 24.9%            |           227 |             5.1 | £1,978         | £2,878,426      | 16.9%           |
|         3 | Recent Low-Value   |        1246 | 21.3%            |            28 |             3   | £841           | £1,048,419      | 6.1%            |
|         2 | Dormant Low-Value  |        1969 | 33.6%            |           393 |             1.4 | £317           | £624,376        | 3.7%            |

## k-selection metrics

|   k |   inertia |   silhouette |
|----:|----------:|-------------:|
|   2 |      8562 |        0.438 |
|   3 |      6338 |        0.347 |
|   4 |      4908 |        0.365 |
|   5 |      4090 |        0.343 |
|   6 |      3549 |        0.335 |
|   7 |      3182 |        0.303 |
|   8 |      2892 |        0.295 |
|   9 |      2650 |        0.289 |
|  10 |      2460 |        0.289 |

## Charts

- `reports/rfm_segments_revenue.html`
- `reports/kmeans_elbow.html`
- `reports/kmeans_silhouette.html`
- `reports/kmeans_scatter.html`

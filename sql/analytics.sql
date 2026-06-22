-- Analytical queries over the consumer-intelligence store.
-- These are the kind of ad-hoc questions a data/ML engineer answers in SQL.
-- All ANSI-compatible (run on PostgreSQL and SQLite).

-- 1. Segment value rollup: who drives revenue and future value?
SELECT segment,
       COUNT(*)                                              AS customers,
       ROUND(SUM(monetary)::numeric, 0)                      AS total_revenue,
       ROUND(AVG(predicted_clv)::numeric, 2)                 AS avg_predicted_clv,
       ROUND(AVG(prob_alive)::numeric, 3)                    AS avg_prob_alive
FROM customers
GROUP BY segment
ORDER BY total_revenue DESC;

-- 2. Revenue concentration (Pareto): share held by the top CLV decile.
WITH ranked AS (
    SELECT predicted_clv,
           NTILE(10) OVER (ORDER BY predicted_clv DESC) AS decile
    FROM customers
)
SELECT decile,
       COUNT(*)                                  AS customers,
       ROUND(SUM(predicted_clv)::numeric, 0)     AS clv_in_decile,
       ROUND(100.0 * SUM(predicted_clv)
             / SUM(SUM(predicted_clv)) OVER (), 1) AS pct_of_total_clv
FROM ranked
GROUP BY decile
ORDER BY decile;

-- 3. High-value customers at churn risk: valuable but unlikely to be alive.
SELECT customer_id, segment, predicted_clv, prob_alive, propensity
FROM customers
WHERE predicted_clv > (SELECT AVG(predicted_clv) FROM customers)
  AND prob_alive < 0.5
ORDER BY predicted_clv DESC
LIMIT 50;

-- 4. Strongest cross-sell rules (single-product antecedent) by lift.
SELECT antecedents, consequents, support, confidence, lift
FROM rules
WHERE antecedents_codes NOT LIKE '%,%'   -- single-item antecedent
ORDER BY lift DESC
LIMIT 25;

-- 5. Targeting list: high propensity to buy AND high predicted value.
SELECT customer_id, segment, propensity, predicted_clv
FROM customers
WHERE propensity > 0.7
ORDER BY predicted_clv DESC
LIMIT 100;

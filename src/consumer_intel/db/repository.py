"""Read queries against the analytics database.

These functions hold the actual SQL the API serves. The SQL is kept ANSI-
compatible so it runs unchanged on both PostgreSQL (production) and SQLite
(local/tests). Parameters are always bound (never string-formatted) to avoid
SQL injection.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session


def get_customer(session: Session, customer_id: str) -> dict | None:
    """Full intelligence profile for one customer, or None if unknown."""
    row = (
        session.execute(
            text(
                """
            SELECT customer_id, recency, frequency, monetary, rfm_score,
                   segment, action, cluster_name, predicted_clv, prob_alive,
                   predicted_purchases, historical_clv, clv_method, propensity
            FROM customers
            WHERE customer_id = :cid
            """
            ),
            {"cid": customer_id},
        )
        .mappings()
        .first()
    )
    return dict(row) if row else None


def segment_summary(session: Session) -> list[dict]:
    """Per-segment rollup: customers, revenue and average predicted CLV."""
    rows = (
        session.execute(
            text(
                """
            SELECT segment,
                   COUNT(*)                AS customers,
                   SUM(monetary)           AS total_revenue,
                   AVG(monetary)           AS avg_monetary,
                   AVG(predicted_clv)      AS avg_predicted_clv,
                   AVG(prob_alive)         AS avg_prob_alive
            FROM customers
            GROUP BY segment
            ORDER BY total_revenue DESC
            """
            )
        )
        .mappings()
        .all()
    )
    return [dict(r) for r in rows]


def top_customers_by_clv(session: Session, limit: int = 20) -> list[dict]:
    """Highest predicted-CLV customers."""
    rows = (
        session.execute(
            text(
                """
            SELECT customer_id, segment, predicted_clv, prob_alive, propensity
            FROM customers
            ORDER BY predicted_clv DESC
            LIMIT :lim
            """
            ),
            {"lim": limit},
        )
        .mappings()
        .all()
    )
    return [dict(r) for r in rows]


def next_best_offers(session: Session, stock_code: str, limit: int = 5) -> list[dict]:
    """Top association rules whose antecedent is exactly this single product.

    Matches rules where the antecedent code set is the one product, ranked by
    lift — the simple single-item Next Best Offer.
    """
    rows = (
        session.execute(
            text(
                """
            SELECT antecedents, consequents, consequents_codes,
                   support, confidence, lift
            FROM rules
            WHERE antecedents_codes = :code
            ORDER BY lift DESC
            LIMIT :lim
            """
            ),
            {"code": stock_code, "lim": limit},
        )
        .mappings()
        .all()
    )
    return [dict(r) for r in rows]


def next_best_offers_for_customer(session: Session, customer_id: str, limit: int = 5) -> list[dict]:
    """Cross-sell recommendations for a customer, based on their top-revenue product.

    Joins ``customer_top_product`` (one representative product per customer)
    to ``rules`` on that product's antecedent — the per-customer counterpart
    to :func:`next_best_offers`, which takes a product code directly.
    """
    rows = (
        session.execute(
            text(
                """
            SELECT r.antecedents, r.consequents, r.consequents_codes,
                   r.support, r.confidence, r.lift
            FROM customer_top_product ctp
            JOIN rules r ON r.antecedents_codes = ctp.stock_code
            WHERE ctp.customer_id = :cid
            ORDER BY r.lift DESC
            LIMIT :lim
            """
            ),
            {"cid": customer_id, "lim": limit},
        )
        .mappings()
        .all()
    )
    return [dict(r) for r in rows]


WIN_BACK_SEGMENTS = ("At Risk", "Can't Lose Them")


def win_back_candidates(session: Session) -> list[dict]:
    """High-value, sleeping customers — the win-back campaign candidate list.

    Segments chosen match Phase 1's finding: At Risk + Can't Lose Them are
    valuable customers who have gone quiet.
    """
    rows = (
        session.execute(
            text(
                """
            SELECT customer_id, segment, recency, frequency, monetary,
                   predicted_clv, prob_alive
            FROM customers
            WHERE segment IN (:seg1, :seg2)
            ORDER BY predicted_clv DESC
            """
            ),
            {"seg1": WIN_BACK_SEGMENTS[0], "seg2": WIN_BACK_SEGMENTS[1]},
        )
        .mappings()
        .all()
    )
    return [dict(r) for r in rows]


def customer_exists(session: Session, customer_id: str) -> bool:
    """Cheap existence check for a customer_id.

    Used by the Copilot graph's router to short-circuit to a not-found
    response before the (more expensive, and partly redundant) fan-out
    queries run.
    """
    return (
        session.execute(
            text("SELECT 1 FROM customers WHERE customer_id = :cid LIMIT 1"),
            {"cid": customer_id},
        ).first()
        is not None
    )


def count_customers(session: Session) -> int:
    """Total customers in the store (health/debug)."""
    return int(session.execute(text("SELECT COUNT(*) FROM customers")).scalar_one())


def list_customers(session: Session, limit: int = 100) -> list[dict]:
    """Browseable customer list (highest spend first) for finding an ID."""
    rows = (
        session.execute(
            text(
                """
            SELECT customer_id, segment, recency, frequency, monetary,
                   predicted_clv, propensity
            FROM customers
            ORDER BY monetary DESC
            LIMIT :lim
            """
            ),
            {"lim": limit},
        )
        .mappings()
        .all()
    )
    return [dict(r) for r in rows]


def top_products(session: Session, limit: int = 20) -> list[dict]:
    """Highest-revenue products (also serves as the product browse list)."""
    rows = (
        session.execute(
            text(
                """
            SELECT stock_code, description, revenue, quantity, orders, customers
            FROM products
            ORDER BY revenue DESC
            LIMIT :lim
            """
            ),
            {"lim": limit},
        )
        .mappings()
        .all()
    )
    return [dict(r) for r in rows]


def get_product(session: Session, stock_code: str) -> dict | None:
    """Single product summary, or None if unknown."""
    row = (
        session.execute(
            text(
                """
            SELECT stock_code, description, revenue, quantity, orders, customers
            FROM products
            WHERE stock_code = :code
            """
            ),
            {"code": stock_code},
        )
        .mappings()
        .first()
    )
    return dict(row) if row else None


def monthly_series(session: Session) -> list[dict]:
    """Month-by-month revenue / orders / customers, chronological."""
    rows = (
        session.execute(
            text("SELECT month, revenue, orders, customers FROM monthly ORDER BY month")
        )
        .mappings()
        .all()
    )
    return [dict(r) for r in rows]


def country_summary(session: Session, limit: int = 15) -> list[dict]:
    """Top countries by revenue."""
    rows = (
        session.execute(
            text(
                """
                SELECT country, revenue, orders, customers
                FROM country
                ORDER BY revenue DESC
                LIMIT :lim
                """
            ),
            {"lim": limit},
        )
        .mappings()
        .all()
    )
    return [dict(r) for r in rows]


def product_overview(session: Session) -> dict:
    """Totals across all products (count, revenue, units) for the KPI row."""
    row = (
        session.execute(
            text(
                """
                SELECT COUNT(*)      AS products,
                       SUM(revenue)   AS revenue,
                       SUM(quantity)  AS quantity
                FROM products
                """
            )
        )
        .mappings()
        .first()
    )
    return dict(row)

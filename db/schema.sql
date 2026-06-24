-- Canonical PostgreSQL schema for the consumer-intelligence store.
-- The loader (db/loader.py) creates these tables automatically via pandas;
-- this file documents the production schema and is used to initialise the
-- Postgres container in docker-compose.

CREATE TABLE IF NOT EXISTS customers (
    customer_id          TEXT PRIMARY KEY,
    recency              INTEGER,
    frequency            INTEGER,
    monetary             DOUBLE PRECISION,
    rfm_score            TEXT,
    segment              TEXT,
    action               TEXT,
    cluster_name         TEXT,
    predicted_clv        DOUBLE PRECISION,
    prob_alive           DOUBLE PRECISION,
    predicted_purchases  DOUBLE PRECISION,
    historical_clv       DOUBLE PRECISION,
    clv_method           TEXT,
    propensity           DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_customers_segment ON customers (segment);
CREATE INDEX IF NOT EXISTS idx_customers_clv ON customers (predicted_clv DESC);

CREATE TABLE IF NOT EXISTS rules (
    antecedents        TEXT,
    consequents        TEXT,
    antecedents_codes  TEXT,
    consequents_codes  TEXT,
    support            DOUBLE PRECISION,
    confidence         DOUBLE PRECISION,
    lift               DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_rules_antecedents ON rules (antecedents_codes);

CREATE TABLE IF NOT EXISTS products (
    stock_code   TEXT PRIMARY KEY,
    description  TEXT,
    revenue      DOUBLE PRECISION,
    quantity     BIGINT,
    orders       BIGINT,
    customers    BIGINT
);

CREATE INDEX IF NOT EXISTS idx_products_revenue ON products (revenue DESC);

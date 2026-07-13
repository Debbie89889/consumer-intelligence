"""Benchmark the customer-insight graph's parallel fetch fan-out against a serial baseline.

Both variants do the *same* four DB calls (get_customer x3 + next_best_offers_
for_customer) and the same LCEL narration step; the only difference is
whether the four fetch calls run one after another (serial baseline) or
concurrently via the compiled LangGraph (parallel). Reports p50/p95 wall-clock
latency per customer, in milliseconds — real measurements, not estimates.

    python scripts/benchmark_copilot_graph.py [--n 30]

Uses DATABASE_URL if set, else the local SQLite dev db (run
`python scripts/load_db.py` first so it's populated).
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

from sqlalchemy.orm import sessionmaker

# Force the deterministic template narrator, even if OPENAI_API_KEY/
# ANTHROPIC_API_KEY happen to be set in this shell. This benchmark measures
# the DB fetch fan-out, not LLM API latency — and doing otherwise would
# spend real API calls on every run.
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("ANTHROPIC_API_KEY", None)

from consumer_intel.copilot_graph.graph import build_customer_insight_graph
from consumer_intel.copilot_graph.nodes import answer_template, build_chat_facts
from consumer_intel.copilot_graph.state import initial_state
from consumer_intel.db import repository
from consumer_intel.db.engine import database_url, make_engine


def _percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    idx = min(len(values) - 1, int(round(pct / 100 * (len(values) - 1))))
    return values[idx]


def run_serial(session_factory: sessionmaker, customer_id: str) -> None:
    """Same four DB calls the graph's fetch nodes make, run one after another."""
    with session_factory() as session:
        rfm = repository.get_customer(session, customer_id)
    with session_factory() as session:
        repository.get_customer(session, customer_id)  # clv (same query, see nodes.py)
    with session_factory() as session:
        repository.get_customer(session, customer_id)  # propensity (same query)
    with session_factory() as session:
        nbo = repository.next_best_offers_for_customer(session, customer_id)
    facts = build_chat_facts(customer_id, {"rfm": rfm, "nbo": nbo})
    answer_template(facts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="number of customers to sample")
    args = parser.parse_args()

    engine = make_engine(database_url())
    session_factory = sessionmaker(bind=engine)
    graph = build_customer_insight_graph(session_factory)

    with session_factory() as session:
        customer_ids = [r["customer_id"] for r in repository.list_customers(session, args.n)]
    if not customer_ids:
        raise SystemExit("No customers in the database — run scripts/load_db.py first.")

    # Warm up (import/JIT/connection-pool warm-up outside the timed loop).
    run_serial(session_factory, customer_ids[0])
    graph.invoke(initial_state("bench", customer_id=customer_ids[0]))

    serial_ms: list[float] = []
    for cid in customer_ids:
        t0 = time.perf_counter()
        run_serial(session_factory, cid)
        serial_ms.append((time.perf_counter() - t0) * 1000)

    parallel_ms: list[float] = []
    for cid in customer_ids:
        t0 = time.perf_counter()
        graph.invoke(initial_state("bench", customer_id=cid))
        parallel_ms.append((time.perf_counter() - t0) * 1000)

    print(f"n = {len(customer_ids)} customers, db = {database_url()}")
    print()
    print(f"{'':10}{'p50 (ms)':>12}{'p95 (ms)':>12}{'mean (ms)':>12}")
    for label, values in (("serial", serial_ms), ("parallel", parallel_ms)):
        print(
            f"{label:10}{_percentile(values, 50):>12.2f}{_percentile(values, 95):>12.2f}"
            f"{statistics.mean(values):>12.2f}"
        )


if __name__ == "__main__":
    main()

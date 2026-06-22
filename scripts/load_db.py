"""Load the phase 1-4 outputs into the analytics database.

    python scripts/load_db.py

Uses DATABASE_URL if set (PostgreSQL in docker-compose), else a local SQLite
file under data/. Run the phase pipelines first so the parquet outputs exist.
"""

from __future__ import annotations

from consumer_intel.db.engine import database_url, make_engine
from consumer_intel.db.loader import load_all


def main() -> None:
    url = database_url()
    print(f"Loading into: {url}")
    engine = make_engine(url)
    counts = load_all(engine)
    for table, n in counts.items():
        print(f"  {table}: {n:,} rows")
    print("Done.")


if __name__ == "__main__":
    main()

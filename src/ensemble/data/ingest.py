"""M0 ingestion: raw CSV → typed DuckDB tables, with integrity checks.

Run: python -m ensemble.data.ingest

Storage choices (D-007):
- ``article_id`` is stored as INTEGER (max 959,461,001 fits in int32) and
  zero-padded to 10 characters only when written to a submission.
- ``customer_id`` (a 64-char hash) is mapped to a dense ``customer_idx``
  INTEGER, which cuts the transactions table by roughly 2 GB. The mapping is
  kept in ``customers``.
- Transactions are stored sorted by date so DuckDB's min/max zone maps make
  date-cutoff filters cheap.
"""
from __future__ import annotations

import json
import time

from ensemble.config import Config, load_config
from ensemble.db import connect

# Wearable slots for outfit completion (DESIGN §5.1). Groups not listed are
# excluded from Track B.
SLOTS = {
    "Garment Upper body": "upper",
    "Garment Lower body": "lower",
    "Garment Full body": "full",
    "Shoes": "shoes",
    "Accessories": "accessories",
    "Socks & Tights": "socks",
    "Swimwear": "swimwear",
}
JEWELLERY_TYPES = ("Earring", "Necklace", "Ring", "Bracelet")


def _csv(cfg: Config, name: str) -> str:
    return str(cfg.path("raw") / name)


def ingest(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    con = connect(cfg)
    t0 = time.time()

    slot_case = " ".join(f"WHEN '{g}' THEN '{s}'" for g, s in SLOTS.items())
    jewellery = ", ".join(f"'{t}'" for t in JEWELLERY_TYPES)
    con.execute(f"""
        CREATE OR REPLACE TABLE articles AS
        SELECT * REPLACE (CAST(article_id AS INTEGER) AS article_id),
               CASE product_group_name {slot_case} END AS slot,
               product_type_name IN ({jewellery}) AS is_jewellery
        FROM read_csv('{_csv(cfg, "articles.csv")}', header=true,
                      types={{'article_id': 'VARCHAR'}})
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE customers AS
        SELECT CAST(row_number() OVER (ORDER BY customer_id) - 1 AS INTEGER) AS customer_idx, *
        FROM read_csv('{_csv(cfg, "customers.csv")}', header=true)
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE transactions AS
        SELECT CAST(t.t_dat AS DATE) AS t_dat,
               c.customer_idx,
               CAST(t.article_id AS INTEGER) AS article_id,
               CAST(t.price AS FLOAT) AS price,
               CAST(t.sales_channel_id AS TINYINT) AS sales_channel_id
        FROM read_csv('{_csv(cfg, "transactions_train.csv")}', header=true,
                      types={{'article_id': 'VARCHAR', 'customer_id': 'VARCHAR'}}) t
        LEFT JOIN customers c USING (customer_id)
        ORDER BY t_dat
    """)

    report = integrity_report(cfg, con)
    report["seconds"] = round(time.time() - t0, 1)
    out = cfg.path("reports") / "m0_integrity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    con.close()
    return report


def integrity_report(cfg: Config, con) -> dict:
    """The checks listed in DATA.md "Integrity checks"."""
    q = lambda sql: con.execute(sql).fetchone()[0]  # noqa: E731
    src = {
        name: q(f"SELECT count(*) FROM read_csv('{_csv(cfg, name + '.csv')}', header=true)")
        for name in ("transactions_train", "customers", "articles")
    }
    missing_days = [
        r[0] for r in con.execute("""
            SELECT d::DATE FROM generate_series(
                (SELECT min(t_dat) FROM transactions),
                (SELECT max(t_dat) FROM transactions), INTERVAL 1 DAY) g(d)
            WHERE d::DATE NOT IN (SELECT DISTINCT t_dat FROM transactions)
        """).fetchall()
    ]
    return {
        "source_rows": src,
        "db_rows": {
            "transactions_train": q("SELECT count(*) FROM transactions"),
            "customers": q("SELECT count(*) FROM customers"),
            "articles": q("SELECT count(*) FROM articles"),
        },
        "orphan_articles": q("""SELECT count(*) FROM transactions t
                                ANTI JOIN articles a USING (article_id)"""),
        "orphan_customers": q("SELECT count(*) FROM transactions WHERE customer_idx IS NULL"),
        "duplicate_article_ids": q("SELECT count(*) - count(DISTINCT article_id) FROM articles"),
        "min_date": q("SELECT min(t_dat) FROM transactions"),
        "max_date": q("SELECT max(t_dat) FROM transactions"),
        "missing_days": missing_days,
    }


if __name__ == "__main__":
    print(json.dumps(ingest(), indent=2, default=str))

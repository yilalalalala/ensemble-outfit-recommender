"""Per-week Track B artifacts shared by the tower process and the ranker process.

LightGBM and PyTorch ship separate OpenMP runtimes and crash when both are loaded
into one process on macOS (see ``ensemble.api.build``). The two-tower stage
therefore runs as its own process (``python -m ensemble.completion.towers``) and
hands its results to the rolling backtest through the cache directory below.
Everything in this module is DuckDB-only, so both processes can call it.

The cache key covers every setting that changes the mined evidence, the eligible
catalogue, the candidate union or the towers, so a stale artifact can never be
picked up after a configuration change.
"""
from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble.completion import association as A
from ensemble.completion import candidates as C
from ensemble.completion import features as FE
from ensemble.completion import protocol as P
from ensemble.data.splits import Week

CACHE_FIELDS = ("window_weeks", "universe_weeks", "pop_days", "min_support", "half_lives_days",
                "min_basket_size", "max_basket_size", "basket_filter", "candidates", "train_queries",
                "neg_sample_rate", "use_clip", "two_tower", "customer_weeks")


def cache_key(cfg) -> str:
    b = dict(cfg.track_b)
    keep = {k: b[k] for k in CACHE_FIELDS if k in b}
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:10]


def cache_dir(cfg) -> Path:
    d = cfg.path("interim") / "track_b" / cache_key(cfg)
    d.mkdir(parents=True, exist_ok=True)
    return d


def tower_paths(cfg, week: Week) -> dict[str, Path]:
    d = cache_dir(cfg)
    return {"tt": d / f"tt_{week.start}.parquet", "ttc": d / f"ttc_{week.start}.parquet",
            "key_sims": d / f"keysims_{week.start}.parquet", "info": d / f"tower_{week.start}.json"}


def price_tiers(con, week: Week, n_tiers: int = 5) -> dict[int, int]:
    """Per-slot price quintile of each article's mean pre-cutoff price (1..5)."""
    rows = con.execute(f"""
        WITH p AS (SELECT article_id, avg(price) AS price FROM transactions
                   WHERE t_dat BETWEEN DATE '{week.start - timedelta(weeks=52)}'
                                 AND DATE '{week.cutoff}' GROUP BY 1)
        SELECT p.article_id, ntile({n_tiers}) OVER (PARTITION BY a.slot ORDER BY p.price)
        FROM p JOIN articles a USING (article_id) WHERE a.slot IS NOT NULL""").fetchall()
    return dict(rows)


def load_clip(cfg):
    """Frozen FashionCLIP image vectors (M7a), or None when disabled or not built."""
    if not cfg.track_b.get("use_clip", True):
        return None
    d = cfg.path("processed") / "clip"
    if not (d / "article_emb.npy").exists():
        return None
    return np.load(d / "article_ids.npy"), np.load(d / "article_emb.npy").astype(np.float32)


def mining_pairs(con, holdout_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Directed cross-slot training pairs from ``tb_bk``, split train / tower-holdout by basket age."""
    pos = con.execute("""
        SELECT x.article_id AS src, y.article_id AS dst, y.slot AS dst_slot, x.age_days
        FROM tb_bk x JOIN tb_bk y
          ON x.customer_idx = y.customer_idx AND x.t_dat = y.t_dat AND x.slot <> y.slot""").df()
    return (pos[pos.age_days >= holdout_days].reset_index(drop=True),
            pos[pos.age_days < holdout_days].reset_index(drop=True))


def prepare_sql(con, cfg, week: Week) -> dict:
    """Mine evidence, build the eligible catalogue and the query set for ``week``.

    Reads nothing dated on or after ``week.start`` apart from the label column.
    """
    funnel = A.mine(con, week, cfg, cfg.track_b.get("basket_filter", ""))
    audit = A.basket_audit(con, week, cfg)
    uni = P.eligible_universe(con, week, cfg)
    q = P.restrict_truth(P.queries(con, week, cfg), uni)
    C.register(con, q, uni)
    FE.register_attrs(con)
    return {"funnel": funnel, "basket_audit": audit, "uni": uni, "q": q,
            "n_truth_dropped": int(q.attrs.get("n_truth_dropped", 0))}


def key_candidates(con) -> pd.DataFrame:
    """Distinct (anchor, target_slot, article_id) over all sources — the unit for pair features."""
    return con.execute("""
        SELECT k.anchor, k.target_slot, a.dst AS article_id
        FROM _tb_keys k JOIN _tb_assoc a ON a.src = k.anchor AND a.dst_slot = k.target_slot
        UNION
        SELECT k.anchor, k.target_slot, y.dst FROM _tb_keys k
             JOIN _tb_attrs ar ON ar.article_id = k.anchor
             JOIN _tb_style y ON y.src_code = ar.product_code AND y.dst_slot = k.target_slot
        UNION
        SELECT k.anchor, k.target_slot, t.dst FROM _tb_keys k
             JOIN _tb_tt t ON t.anchor = k.anchor AND t.target_slot = k.target_slot
        UNION
        SELECT k.anchor, k.target_slot, c.dst FROM _tb_keys k
             JOIN _tb_ttc c ON c.anchor = k.anchor AND c.target_slot = k.target_slot
        UNION
        SELECT k.anchor, k.target_slot, p.dst FROM _tb_keys k
             JOIN _tb_pop p ON p.target_slot = k.target_slot
        ORDER BY 1, 2, 3""").df()

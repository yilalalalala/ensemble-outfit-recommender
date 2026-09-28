"""M7a for Track A: FashionCLIP-based retrieval channel and ranker features.

Uses only numpy (Accelerate BLAS), never PyTorch: this code runs in the same
process as LightGBM, and loading both OpenMP runtimes deadlocks on macOS.

- ``build_visual_channel``: for each customer, a profile vector (mean embedding of
  their most recent purchases) is matched against the recent assortment (articles
  sold in the last ``visual_pool_days``). Writes TEMP TABLE ``_visual``.
- ``add_clip_features``: cosine similarity between each candidate and the
  customer's recent purchases (max and mean).

Both read only transactions before ``week.start`` (D-003).
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from ensemble.data.splits import Week

_cache: dict = {}


def clip_matrix(cfg):
    if "emb" not in _cache:
        d = cfg.path("processed") / "clip"
        ids = np.load(d / "article_ids.npy")
        _cache["emb"] = np.load(d / "article_emb.npy").astype(np.float32)
        _cache["row"] = pd.Series(np.arange(len(ids)), index=ids)
        _cache["has"] = np.load(d / "has_image.npy")
    return _cache["emb"], _cache["row"], _cache["has"]


def _recent_items(con, week: Week, users_table: str, n: int) -> pd.DataFrame:
    return con.execute(f"""
        SELECT customer_idx, article_id FROM (
            SELECT t.customer_idx, t.article_id, max(t.t_dat) AS last_dat
            FROM transactions t JOIN {users_table} u USING (customer_idx)
            WHERE t.t_dat < DATE '{week.start}' AND t.t_dat >= DATE '{week.start - timedelta(weeks=26)}'
            GROUP BY 1, 2)
        QUALIFY row_number() OVER (PARTITION BY customer_idx ORDER BY last_dat DESC, article_id) <= {int(n)}
    """).df()


def build_visual_channel(con, week: Week, cfg, users_table: str = "_users") -> None:
    r = cfg.retrieval
    emb, row, has = clip_matrix(cfg)
    hist = _recent_items(con, week, users_table, int(r.visual_profile_items))
    hist = hist[has[row.reindex(hist.article_id).fillna(-1).astype(int).clip(lower=0).values]
                & hist.article_id.isin(row.index)]
    pool = con.execute(f"""SELECT DISTINCT article_id FROM transactions
                           WHERE t_dat >= DATE '{week.start - timedelta(days=int(r.visual_pool_days))}'
                             AND t_dat < DATE '{week.start}'""").fetchnumpy()["article_id"]
    pool = pool[has[row.reindex(pool).fillna(0).astype(int).values] & np.isin(pool, row.index)]
    P = emb[row[pool].values]                                          # pool × d
    users, codes = np.unique(hist.customer_idx.values, return_inverse=True)
    prof = np.zeros((len(users), emb.shape[1]), dtype=np.float32)
    np.add.at(prof, codes, emb[row[hist.article_id].values])
    prof /= np.linalg.norm(prof, axis=1, keepdims=True) + 1e-9
    k = int(r.visual_k)
    out_u, out_a, out_s = [], [], []
    for b in range(0, len(users), 4096):
        S = prof[b:b + 4096] @ P.T
        top = np.argpartition(-S, k, axis=1)[:, :k]
        sc = np.take_along_axis(S, top, axis=1)
        out_u.append(np.repeat(users[b:b + 4096], k))
        out_a.append(pool[top].ravel())
        out_s.append(sc.ravel())
    vis = pd.DataFrame({"customer_idx": np.concatenate(out_u) if out_u else [],
                        "article_id": np.concatenate(out_a) if out_a else [],
                        "score": np.concatenate(out_s) if out_s else []})
    con.execute("CREATE OR REPLACE TEMP TABLE _visual AS SELECT * FROM vis")


def add_clip_features(con, week: Week, cfg, df: pd.DataFrame) -> pd.DataFrame:
    emb, row, has = clip_matrix(cfg)
    con.execute("CREATE OR REPLACE TEMP TABLE _clipu AS SELECT DISTINCT customer_idx FROM cand")
    hist = _recent_items(con, week, "_clipu", int(cfg.retrieval.visual_profile_items))
    hist = hist[hist.article_id.isin(row.index)]
    hist = hist[has[row[hist.article_id].values]]
    groups = {c: emb[row[g.article_id].values] for c, g in hist.groupby("customer_idx")}
    cand_rows = row.reindex(df.article_id.values).values
    ok = ~np.isnan(cand_rows)
    cand_emb = np.zeros((len(df), emb.shape[1]), dtype=np.float32)
    cand_emb[ok] = emb[cand_rows[ok].astype(int)]
    cand_has = np.zeros(len(df), dtype=bool)
    cand_has[ok] = has[cand_rows[ok].astype(int)]
    mx = np.full(len(df), np.nan, dtype=np.float32)
    mean = np.full(len(df), np.nan, dtype=np.float32)
    cust = df.customer_idx.values
    bounds = np.flatnonzero(np.r_[True, cust[1:] != cust[:-1], True])
    for s, e in zip(bounds[:-1], bounds[1:]):
        H = groups.get(cust[s])
        if H is None:
            continue
        S = cand_emb[s:e] @ H.T
        mx[s:e], mean[s:e] = S.max(axis=1), S.mean(axis=1)
    mx[~cand_has], mean[~cand_has] = np.nan, np.nan
    return df.assign(ca_clip_sim_max=mx, ca_clip_sim_mean=mean)

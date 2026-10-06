"""Collaborative latent-factor retrieval channel: implicit ALS (Hu, Koren & Volinsky 2008).

The user × item matrix holds every customer's purchases in the ``als_weeks``
before the cutoff, with confidence ``1 + als_alpha · log1p(count)`` (count is
recency-weighted when ``als_recency``). Items need ``als_min_item_count``
purchases in the window. Factors are fitted in a **subprocess**: ``implicit``
and LightGBM bundle different OpenMP runtimes, which deadlock when loaded into
one process on macOS (HANDOFF §6). The fit is cached per cutoff and parameter
set under ``data/interim/als/``.

Scoring is a numpy dot product of the target customers' factors against the
factors of the live pool (articles sold in the ``als_pool_days`` before the
cutoff); the top ``als_k`` become candidates. Customers without factors (no
purchase in the window) get none.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble.data.splits import Week

PARAMS = ("als_weeks", "als_factors", "als_iterations", "als_regularization", "als_alpha",
          "als_min_item_count", "als_recency", "als_seed")


def _cache_path(cfg, week: Week, r) -> Path:
    key = json.dumps({p: r.get(p) for p in PARAMS}, sort_keys=True, default=str)
    h = hashlib.sha1(f"{week.start}|{key}".encode()).hexdigest()[:12]
    d = cfg.path("interim") / "als"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"als_{week.start}_{h}"


def fit_factors(con, week: Week, r) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (user_ids, user_factors, item_ids, item_factors), fitting in a subprocess if not cached."""
    from ensemble.config import load_config
    base = _cache_path(load_config(), week, r)
    out = base.with_suffix(".factors.npz")
    if not out.exists():
        start = f"DATE '{week.start}'"
        lo = f"DATE '{week.start - timedelta(weeks=int(r.als_weeks))}'"
        w = (f"1.0 / (1 + date_diff('day', t_dat, {start}) / 7.0)" if r.get("als_recency", True) else "1.0")
        df = con.execute(f"""
            WITH x AS (SELECT customer_idx, article_id, sum({w}) AS c FROM transactions
                       WHERE t_dat >= {lo} AND t_dat < {start} GROUP BY 1, 2),
            items AS (SELECT article_id FROM x GROUP BY 1 HAVING count(*) >= {int(r.als_min_item_count)})
            SELECT customer_idx, article_id, c FROM x JOIN items USING (article_id)
        """).df()
        users, ui = np.unique(df.customer_idx.to_numpy(), return_inverse=True)
        items, ii = np.unique(df.article_id.to_numpy(), return_inverse=True)
        conf = 1.0 + float(r.als_alpha) * np.log1p(df.c.to_numpy())
        inp = base.with_suffix(".input.npz")
        np.savez(inp, users=users, items=items, row=ui.astype(np.int32), col=ii.astype(np.int32),
                 data=conf.astype(np.float32))
        args = json.dumps({p: r.get(p) for p in PARAMS})
        subprocess.run([sys.executable, "-m", "ensemble.candidates.als", str(inp), str(out), args], check=True)
        inp.unlink()
    z = np.load(out)
    return z["users"], z["user_factors"], z["items"], z["item_factors"]


def build_als_channel(con, week: Week, r, users_table: str) -> None:
    users, U, items, V = fit_factors(con, week, r)
    pool = con.execute(f"""SELECT DISTINCT article_id FROM transactions
                           WHERE t_dat >= DATE '{week.start - timedelta(days=int(r.als_pool_days))}'
                             AND t_dat < DATE '{week.start}'""").fetchnumpy()["article_id"]
    item_pos = pd.Series(np.arange(len(items)), index=items)
    pool = np.sort(pool[np.isin(pool, items)])
    P = V[item_pos[pool].to_numpy()]
    targets = con.execute(f"SELECT customer_idx FROM {users_table}").fetchnumpy()["customer_idx"]
    user_pos = pd.Series(np.arange(len(users)), index=users)
    targets = np.sort(targets[np.isin(targets, users)])
    k = min(int(r.als_k), len(pool))
    out_u, out_a, out_s = [], [], []
    for b in range(0, len(targets), 8192):
        tb = targets[b:b + 8192]
        S = U[user_pos[tb].to_numpy()] @ P.T
        top = np.argpartition(-S, k - 1, axis=1)[:, :k]
        out_u.append(np.repeat(tb, k))
        out_a.append(pool[top].ravel())
        out_s.append(np.take_along_axis(S, top, axis=1).ravel().astype(np.float64))
    als = pd.DataFrame({"customer_idx": np.concatenate(out_u).astype(np.int32) if out_u else np.array([], np.int32),
                        "article_id": np.concatenate(out_a).astype(np.int32) if out_a else np.array([], np.int32),
                        "score": np.concatenate(out_s) if out_s else np.array([], np.float64)})
    con.register("_als_df", als)
    con.execute("CREATE OR REPLACE TEMP TABLE _ch_als AS SELECT * FROM _als_df")
    con.unregister("_als_df")


def _fit_main(inp: str, out: str, args: str) -> None:
    import os
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import scipy.sparse as sp
    from implicit.cpu.als import AlternatingLeastSquares
    p = json.loads(args)
    z = np.load(inp)
    m = sp.csr_matrix((z["data"], (z["row"], z["col"])), shape=(len(z["users"]), len(z["items"])))
    model = AlternatingLeastSquares(factors=int(p["als_factors"]), regularization=float(p["als_regularization"]),
                                    iterations=int(p["als_iterations"]), random_state=int(p.get("als_seed") or 42),
                                    calculate_training_loss=False)
    model.fit(m, show_progress=False)
    np.savez(out, users=z["users"], items=z["items"], user_factors=np.asarray(model.user_factors, dtype=np.float32),
             item_factors=np.asarray(model.item_factors, dtype=np.float32))


if __name__ == "__main__":
    _fit_main(*sys.argv[1:4])

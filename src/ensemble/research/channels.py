"""The neural baselines as retrieval channels and ranker features for the Track A ensemble.

Both read the per-week cache written by ``ensemble.research.neural``. The cache key contains
the label week, so a week can only ever read the model whose inputs stopped at its own
cutoff. A missing cache is an error, never a silent empty channel.

- **Channel** (``bpr``, ``lightgcn``, ``sasrec`` in ``retrieval.channels``): the model's exact
  top-k over the eligible catalogue, under the standard contract
  ``(customer_idx, article_id, score)``; its rank and score become provenance features.
- **Score features** ``nn_<model>_dot``: the model's dot product for *every* candidate,
  whichever channel proposed it (NaN when the customer or article has no vector).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ensemble.config import load_config
from ensemble.data.splits import Week
from ensemble.research.neural import MODELS, cache_dir, params_for


def model_dir(cfg, model: str, week: Week, seed: int) -> Path:
    if "research" not in cfg:
        raise ValueError("neural channels and features need the research config (ENSEMBLE_CONFIG=track_a_research)")
    d = cache_dir(cfg, model, str(week.start), int(seed), params_for(cfg, model))
    if not (d / "result.json").exists():
        raise FileNotFoundError(f"no cached {model} model for week {week.start} seed {seed} ({d}); "
                                f"build it with `python -m ensemble.research.pipeline models`")
    return d


def build_neural_channel(con, week: Week, r, users_table: str, model: str) -> None:
    cfg = load_config()
    path = model_dir(cfg, model, week, int(r.get("neural_seed", 0))) / "topk.parquet"
    con.execute(f"""CREATE OR REPLACE TEMP TABLE _ch_{model} AS
                    SELECT t.customer_idx, t.article_id, t.score::DOUBLE AS score FROM read_parquet('{path}') t
                    WHERE t.customer_idx IN (SELECT customer_idx FROM {users_table})""")


def add_neural_scores(df: pd.DataFrame, cfg, week: Week, models=MODELS, seed: int = 0,
                      chunk: int = 1_000_000) -> pd.DataFrame:
    """Append ``nn_<model>_dot`` for every (customer, article) row of ``df``."""
    cust = df["customer_idx"].to_numpy(np.int64)
    art = df["article_id"].to_numpy(np.int64)
    for m in models:
        z = np.load(model_dir(cfg, m, week, seed) / "emb.npz")
        users, U, items, V = z["users"], z["user_emb"], z["items"], z["item_emb"]
        out = np.full(len(df), np.nan, dtype=np.float32)
        for b in range(0, len(df), chunk):
            c, a = cust[b:b + chunk], art[b:b + chunk]
            ui = np.searchsorted(users, c)
            ii = np.searchsorted(items, a)
            ok = (ui < len(users)) & (ii < len(items))
            ok[ok] = (users[ui[ok]] == c[ok]) & (items[ii[ok]] == a[ok])
            vals = np.full(len(c), np.nan, dtype=np.float32)
            vals[ok] = np.einsum("ij,ij->i", U[ui[ok]], V[ii[ok]])
            out[b:b + chunk] = vals
        df[f"nn_{m}_dot"] = out
    return df


def _check_registered() -> None:
    from ensemble.candidates.retrieval import _REGISTRY
    missing = [m for m in MODELS if m not in _REGISTRY]
    assert not missing, missing


if __name__ == "__main__":
    _check_registered()
    print(load_config().get("research") is not None)

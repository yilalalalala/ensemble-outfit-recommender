"""Score every single-stage system on the reporting folds under the frozen protocol.

  ENSEMBLE_CONFIG=track_a_research python -m ensemble.research.benchmark

Systems (each restricted to E(w) and back-filled identically, ``evaluate.score_lists``):

- ``popularity_global``      top sellers of the 7 days before the cutoff (D-004 reference)
- ``popularity_age``         the same within the customer's age band
- ``repeat_pop_age``         repeat purchase (4 weeks) + age-band popularity: the M1 best rule
- ``item_cf``                item-to-item CF (cosine co-purchase), the Phase-2 ``cf`` channel alone
- ``covis``                  directional co-visitation, the Phase-2 ``covis`` channel alone
- ``bpr`` / ``lightgcn`` / ``sasrec``   the reproduced baselines, one row per seed

Outputs: ``data/interim/track_a_research/eval/<system>/<fold>_s<seed>.parquet`` (+ ``.json``).
Restartable: existing outputs are skipped. Reads no label week except the one being scored.
"""
from __future__ import annotations

import json

import pandas as pd

from ensemble.baselines import repeat_purchase
from ensemble.config import Config, load_config
from ensemble.db import connect
from ensemble.research import protocol as P
from ensemble.research.ensemble import eval_path
from ensemble.research.evaluate import FoldContext, lists_from_topk, score_lists, summarize
from ensemble.research.neural import MODELS, cache_dir, params_for


def channel_lists(con, week, cfg, ch: str, users) -> dict[int, list[int]]:
    from ensemble.candidates.retrieval import _REGISTRY, build_shared
    con.execute("CREATE OR REPLACE TEMP TABLE _users AS SELECT unnest(?::INTEGER[]) AS customer_idx", [list(users)])
    r = Config(dict(cfg.retrieval))
    build_shared(con, week, r, "_users")
    _REGISTRY[ch](con, week, r, "_users")
    df = con.execute(f"""SELECT customer_idx, article_id FROM _ch_{ch}
                         QUALIFY row_number() OVER (PARTITION BY customer_idx
                                                    ORDER BY round(score::DOUBLE, 9) DESC, article_id) <= 50
                         ORDER BY customer_idx, round(score::DOUBLE, 9) DESC, article_id""").df()
    for t in ("_hist", "_sales", f"_ch_{ch}"):
        con.execute(f"DROP TABLE IF EXISTS {t}")
    return df.groupby("customer_idx", sort=False)["article_id"].agg(list).to_dict()


def save(cfg, system: str, fold, seed: int, res: pd.DataFrame, extra: dict | None = None) -> None:
    out = eval_path(cfg, system, fold, seed)
    res.assign(fold=str(fold.start)).to_parquet(out, index=False)
    meta = {"system": system, "fold": str(fold.start), "seed": seed, "summary": summarize(res),
            "coverage": res.attrs["coverage"], "novelty": res.attrs["novelty"],
            "fallback_only_share": res.attrs["fallback_only_share"], **(extra or {})}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=float))
    print(f"  {system:18s} {fold.start} s{seed}: MAP@12 {meta['summary']['map@12']:.5f}", flush=True)


def run(models=MODELS, seeds=None) -> None:
    cfg = load_config()
    p = P.proto(cfg)
    seeds = list(seeds or p.seeds.neural)
    con = connect(cfg, read_only=True)
    for fold in P.reporting_folds(cfg):
        todo_rules = [s for s in ("popularity_global", "popularity_age", "repeat_pop_age", "item_cf", "covis")
                      if not eval_path(cfg, s, fold, 0).with_suffix(".json").exists()]
        todo_neural = [(m, s) for m in models for s in seeds
                       if not eval_path(cfg, m, fold, s).with_suffix(".json").exists()
                       and (cache_dir(cfg, m, str(fold.start), s, params_for(cfg, m)) / "result.json").exists()]
        if not todo_rules and not todo_neural:
            continue
        print(f"fold {fold.start}", flush=True)
        ctx = FoldContext(con, cfg, fold)
        users = ctx.users
        from ensemble.baselines import popular, popular_by_age
        pop, pop_age = popular(con, fold), popular_by_age(con, fold)
        for s in todo_rules:
            if s == "popularity_global":
                raw = {u: pop for u in users}
            elif s == "popularity_age":
                raw = {u: ctx.fallback(u) for u in users}
            elif s == "repeat_pop_age":
                raw = repeat_purchase(con, fold, users)
            else:
                raw = channel_lists(con, fold, cfg, {"item_cf": "cf", "covis": "covis"}[s], users)
            save(cfg, s, fold, 0, score_lists(ctx, raw))
        del pop_age
        for m, s in todo_neural:
            d = cache_dir(cfg, m, str(fold.start), s, params_for(cfg, m))
            raw = lists_from_topk(pd.read_parquet(d / "topk.parquet"))
            info = json.loads((d / "result.json").read_text())
            res = {k: info.get(k) for k in ("train_seconds", "score_seconds", "wall_seconds", "peak_rss_bytes",
                                             "mps_driver_bytes", "n_parameters", "artifact_bytes", "n_users",
                                             "n_items", "n_interactions", "n_train_sequences", "n_targets",
                                             "n_targets_represented", "epochs_log", "params", "device")}
            save(cfg, m, fold, s, score_lists(ctx, raw), {"model_run": res, "cache_dir": d.name})


if __name__ == "__main__":
    run()

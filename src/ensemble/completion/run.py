"""M4/M5: Track B outfit completion — mining, models, evaluation, ablations.

  python -m ensemble.completion.run val     model selection + ablations on validation
  python -m ensemble.completion.run test    selected configuration, scored once on test
  python -m ensemble.completion.run serve   train on the latest window, save artifacts for serving
"""
from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd
import torch

from ensemble import tracking
from ensemble.completion import models as M
from ensemble.completion.data import item_table, mine_pairs, price_tiers, queries, universe
from ensemble.completion.evaluate import evaluate
from ensemble.config import Config, load_config
from ensemble.data.splits import load_splits
from ensemble.db import connect
from ensemble.evaluation.metrics import relative_lift


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def training_examples(con) -> pd.DataFrame:
    """One row per (basket, directed cross-slot pair) in the mining window."""
    return con.execute("""
        SELECT x.article_id AS src, y.article_id AS dst, y.slot AS dst_slot
        FROM _bk x JOIN _bk y ON x.customer_idx = y.customer_idx AND x.t_dat = y.t_dat AND x.slot <> y.slot""").df()


def setup(con, cfg, week):
    t = time.time()
    funnel = mine_pairs(con, week, cfg)
    uni = universe(con, week, cfg)
    q = queries(con, week, cfg) if week.end <= con.execute("SELECT max(t_dat) FROM transactions").fetchone()[0] else None
    if q is not None:
        con.execute("CREATE OR REPLACE TEMP TABLE _q_anchors AS SELECT DISTINCT anchor FROM q")
    tiers = price_tiers(con, week, cfg)
    items = item_table(con)
    pos = training_examples(con)
    print(f"  mined {len(pos):,} training examples; {len(uni):,} live items; "
          f"{0 if q is None else len(q):,} queries ({time.time() - t:.0f}s)", flush=True)
    return funnel, uni, q, tiers, items, pos


def load_clip(cfg):
    d = cfg.path("processed") / "clip"
    if not (d / "article_emb.npy").exists():
        return None
    return np.load(d / "article_ids.npy"), np.load(d / "article_emb.npy").astype(np.float32)


def fit_tt(cfg, pos, items, tiers, uni, negatives=None, epochs=None, use_clip=False, seed=0):
    tt = Config({**dict(cfg.completion.two_tower)})
    if negatives is not None:
        tt["negatives"] = negatives
    if epochs is not None:
        tt["epochs"] = epochs
    enc = M.ItemEncoder(items, tiers, np.unique(np.r_[pos.src.values, pos.dst.values]), M.pop_buckets(uni),
                        clip=load_clip(cfg) if use_clip else None)
    t = time.time()
    model = M.train_two_tower(pos, enc, uni, tiers, tt, device=device(), seed=seed)
    return model, enc, time.time() - t


def score(name, recs, q, uni, cfg, results, pop_recall=None):
    m = evaluate(recs, q, uni, cfg)
    k = int(cfg.completion.k)
    if pop_recall is not None:
        m[f"relative_lift_recall@{k}_vs_popularity"] = relative_lift(m[f"recall@{k}"], pop_recall)
    results[name] = m
    lift = m.get(f"relative_lift_recall@{k}_vs_popularity")
    print(f"  {name:32s} R@{k}={m[f'recall@{k}']:.4f} NDCG={m[f'ndcg@{k}']:.4f} "
          f"lift={'—' if lift is None else f'{lift:+.1%}'} tail={m[f'recall@{k}_tail']:.4f} "
          f"cold={m[f'recall@{k}_cold']:.4f} jew={m[f'recall@{k}_jewellery']:.4f} "
          f"cov={m[f'catalog_coverage@{k}']:.3f}", flush=True)
    return m


def run(mode: str) -> dict:
    cfg = load_config()
    k = int(cfg.completion.k)
    con = connect(cfg, read_only=True)
    splits = load_splits(con, cfg)
    week = {"val": splits.val, "test": splits.test, "serve": splits.submission}[mode]
    print(f"Track B {mode}: target week {week.start}..{week.end}", flush=True)
    funnel, uni, q, tiers, items, pos = setup(con, cfg, week)
    print("  funnel:", funnel, flush=True)

    if mode == "serve":
        sel = json.loads((cfg.path("reports") / "m4_selected.json").read_text())
        model, enc, secs = fit_tt(cfg, pos, items, tiers, uni, use_clip=sel.get("use_clip", False))
        save_artifacts(cfg, con, model, enc, uni, tiers)
        from ensemble.completion.serve_ctl import main as precompute_ctl
        precompute_ctl()
        return {"funnel": funnel}

    results: dict = {}
    pop = score("popularity", M.popularity(q, uni, k), q, uni, cfg, results)
    p0 = pop[f"recall@{k}"]
    selected = json.loads((cfg.path("reports") / "m4_selected.json").read_text()) if mode == "test" else {}

    # Association rules (market basket analysis).
    supports = [selected["min_support"]] if mode == "test" else [2, 3, 5, 10]
    assoc = {}
    for s in supports:
        assoc[s] = M.association(con, q, uni, k, s, "npmi")
        score(f"association_npmi_support{s}", assoc[s], q, uni, cfg, results, p0)
    best_s = selected.get("min_support") or max(supports, key=lambda s: results[f"association_npmi_support{s}"][f"recall@{k}"])
    score(f"ablation_association_rawcount_support{best_s}", M.association(con, q, uni, k, best_s, "co"), q, uni, cfg, results, p0)

    score(f"association_backoff_support{best_s}", M.association_backoff(con, q, uni, k, best_s), q, uni, cfg, results, p0)

    # Two-tower, with and without FashionCLIP image vectors (DESIGN §5.6 ablation #2).
    tt_lists = {}
    for use_clip in (False, True):
        tag = "two_tower_clip" if use_clip else "two_tower"
        model, enc, secs = fit_tt(cfg, pos, items, tiers, uni, use_clip=use_clip)
        print(f"  {tag} trained in {secs:.0f}s", flush=True)
        tt_lists[use_clip] = M.two_tower_recs(model, enc, q, uni, 4 * k, device())
        score(tag, [r[:k] for r in tt_lists[use_clip]], q, uni, cfg, results, p0)
        score(tag + "_content_only", M.two_tower_recs(model, enc, q, uni, k, device(), use_ids=False), q, uni, cfg, results, p0)

    # Hybrids: weighted reciprocal rank fusion of an association list and a two-tower list.
    assoc_lists = {False: [a[:4 * k] for a in M.association(con, q, uni, 4 * k, best_s, "npmi")],
                   True: [a[:4 * k] for a in M.association_backoff(con, q, uni, 4 * k, best_s)]}
    combos = [(selected["use_backoff"], selected["use_clip"], selected["rrf_weight"])] if mode == "test" and "use_clip" in selected \
        else [(bo, cl, w) for bo in (False, True) for cl in (False, True) for w in (0.3, 0.5, 0.7)]
    if mode == "test":
        combos.append((False, False, 0.5))  # the MVP configuration, for comparison on the same week
    for bo, cl, w in dict.fromkeys(combos):
        name = f"hybrid_rrf{'_backoff' if bo else ''}{'_clip' if cl else ''}_w{w}"
        score(name, M.rrf([assoc_lists[bo], tt_lists[cl]], [w, 1 - w], k), q, uni, cfg, results, p0)
    if mode == "val":
        best = max(combos, key=lambda c: results[f"hybrid_rrf{'_backoff' if c[0] else ''}{'_clip' if c[1] else ''}_w{c[2]}"][f"recall@{k}"])
        best_bo, best_cl, best_w = best
    else:
        best_bo, best_cl, best_w = selected["use_backoff"], selected["use_clip"], selected["rrf_weight"]

    if mode == "val":
        # Ablation: negative sampling (DESIGN §5.6 #3), 1 epoch, 3 seeds each; mean and std reported.
        for negs in (["in_batch"], ["in_batch", "logq"], ["in_batch", "logq", "pop"], ["in_batch", "logq", "pop", "hard"]):
            runs = []
            for seed in (0, 1, 2):
                mdl, e, _ = fit_tt(cfg, pos, items, tiers, uni, negatives=negs, epochs=1, seed=seed)
                runs.append(evaluate(M.two_tower_recs(mdl, e, q, uni, k, device()), q, uni, cfg))
            name = "ablation_negatives_" + "+".join(negs)
            m = {kk: float(np.mean([r[kk] for r in runs])) for kk in runs[0] if isinstance(runs[0][kk], float)}
            m[f"recall@{k}_std"] = float(np.std([r[f"recall@{k}"] for r in runs]))
            m[f"relative_lift_recall@{k}_vs_popularity"] = relative_lift(m[f"recall@{k}"], p0)
            results[name] = m
            print(f"  {name:45s} R@{k}={m[f'recall@{k}']:.4f}±{m[f'recall@{k}_std']:.4f} "
                  f"lift={m[f'relative_lift_recall@{k}_vs_popularity']:+.1%}", flush=True)
        (cfg.path("reports") / "m4_selected.json").write_text(json.dumps(
            {"min_support": best_s, "rrf_weight": best_w, "use_clip": best_cl, "use_backoff": best_bo}))

    sel = {"min_support": best_s, "rrf_weight": best_w, "use_clip": best_cl, "use_backoff": best_bo}
    out = {"funnel": funnel, "selected": sel, "results": results}
    (cfg.path("reports") / f"m4_track_b_{mode}.json").write_text(json.dumps(out, indent=2, default=str))
    with tracking.run("track_b", f"m4/{mode}", {**sel, **dict(cfg.completion.two_tower)}):
        for name, m in results.items():
            tracking.log_metrics({kk: v for kk, v in m.items() if kk.startswith(("recall", "ndcg", "relative", "catalog", "novelty"))},
                                 prefix=f"{name}.")
    return out


def save_artifacts(cfg, con, model, enc, uni, tiers) -> None:
    out = cfg.path("processed") / "models"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "two_tower.pt")
    with open(out / "two_tower_encoder.pkl", "wb") as f:
        pickle.dump({"enc": enc, "tiers": tiers, "tt": dict(cfg.completion.two_tower),
                     "clip_dim": 0 if enc.clip is None else enc.clip.shape[1]}, f)
    uni.to_parquet(out / "completion_universe.parquet")
    con.execute(f"""COPY (SELECT src, dst, dst_slot, co, lift, npmi FROM pairs_all
                          WHERE co >= {int(json.loads((cfg.path('reports') / 'm4_selected.json').read_text())['min_support'])} AND lift > 1)
                    TO '{out / "completion_pairs.parquet"}' (FORMAT parquet)""")
    con.execute(f"""COPY (SELECT src_code, dst, dst_slot, co, lift, npmi FROM style_pairs
                          WHERE co >= {int(json.loads((cfg.path('reports') / 'm4_selected.json').read_text())['min_support'])} AND lift > 1)
                    TO '{out / "completion_style_pairs.parquet"}' (FORMAT parquet)""")


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "val")

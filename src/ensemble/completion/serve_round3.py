"""Serving artifacts for the Round-3 Track B model (Phase 7).

  python -m ensemble.completion.serve_round3 towers   # PyTorch stage (no LightGBM)
  python -m ensemble.completion.serve_round3 build    # LightGBM stage; runs the above if needed

What changes versus ``ensemble.completion.serve_ctl`` (Round 1): the fixed
RRF(w = 0.5) of association and the two towers is replaced by the learned
compatibility ranker, and every row carries the evidence it was actually ranked
on, so a reason chip can never claim support that is not there (D-035).

Why the *compatibility* ranker and not the personalized one. Complete the Look is
an anchor-level surface: it has to answer for an anonymous visitor on any product
page, so the table is precomputed per (anchor, target slot) and cannot hold a row
per customer. ``lgbm_personalized`` measures what an online feature store would
add (+14% relative Recall@12 over ``lgbm_compatibility`` on the rolling folds);
shipping it needs request-time features and a model server, which this local
SQLite demo does not have. The gap is reported, not hidden (D-033).

The serving week is the week *after* the data ends, so every mined pair, every
catalogue statistic and every price tier is legal — nothing is held back. The
ranker is trained on the last ``train_weeks`` label weeks of the dataset, each
with its own point-in-time features.

Memory. Serving covers every live anchor against every other slot — an order of
magnitude more keys than one backtest fold — so both stages work in batches of
``serve_anchor_batch`` anchors and nothing larger than one batch is ever resident.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from ensemble.completion import candidates as C
from ensemble.completion import features as FE
from ensemble.completion import fusion as FU
from ensemble.completion import pipeline as PL
from ensemble.completion import protocol as P
from ensemble.config import ROOT, load_config
from ensemble.db import connect

SHIPPED_MODEL = "lgbm_compatibility"


def _prepare(con, cfg):
    """Mine evidence and build the eligible catalogue and key space for the serving week."""
    from ensemble.completion import association as A
    from ensemble.data.splits import load_splits

    splits = load_splits(con, cfg)
    week = PL.serving_week(splits)
    A.mine(con, week, cfg, cfg.track_b.get("basket_filter", ""))
    uni = P.eligible_universe(con, week, cfg)
    q = PL.serving_queries(uni)
    C.register(con, q, uni)
    FE.register_attrs(con)
    return splits, week, uni, q


def batches(cfg, q: pd.DataFrame) -> list[tuple[int, int]]:
    return PL.serving_batches(q, int(cfg.track_b.get("serve_anchor_batch", 2000)))


# ------------------------------------------------------------------ stage 1: towers (PyTorch)


def build_towers(log=print) -> dict:
    """Train the towers on everything before the serving week and cache retrieval + pair sims."""
    from ensemble.completion import towers as TW
    from ensemble.completion.data import item_table
    from ensemble.completion.models import pop_buckets

    cfg = load_config()
    con = connect(cfg, read_only=True)
    t0 = time.time()
    splits, week, uni, q = _prepare(con, cfg)
    log(f"serving week {week.start}..{week.end} (cutoff {week.cutoff}): "
        f"{len(uni):,} live articles, {len(q):,} (anchor, target slot) keys")
    tr, hold = PL.mining_pairs(con, int(cfg.track_b.two_tower.holdout_days))
    enc = TW.Encoder(item_table(con), PL.price_tiers(con, week),
                     np.unique(np.r_[tr.src.to_numpy(), tr.dst.to_numpy()]), pop_buckets(uni),
                     clip=PL.load_clip(cfg))
    tw, info = TW.train(con, tr, enc, uni, cfg, holdout=hold, log=log)
    paths = PL.serve_paths(cfg)
    cand = cfg.track_b.candidates
    keys = q[["anchor", "target_slot"]].drop_duplicates().reset_index(drop=True)
    tt = tw.retrieve(keys, uni, int(cand["two_tower"]))
    ttc = tw.retrieve(keys, uni, int(cand["two_tower_content"]), use_ids=False)
    tt.to_parquet(paths["tt"], index=False)
    ttc.to_parquet(paths["ttc"], index=False)
    log(f"  retrieval: {len(tt):,} two-tower rows, {len(ttc):,} content-only rows")
    C.build_sources(con, cfg, tt, ttc)
    n_keys_total = 0
    for i, (lo, hi) in enumerate(batches(cfg, q)):
        C.register_keys(con, lo, hi)
        kc = PL.key_candidates(con)
        a, c, s = kc.anchor.to_numpy(), kc.article_id.to_numpy(), kc.target_slot.to_numpy()
        kc["tt_pair_sim"] = tw.pair_similarity(a, c, s, use_ids=True)
        kc["ttc_pair_sim"] = tw.pair_similarity(a, c, s, use_ids=False)
        kc["clip_sim"] = TW.clip_similarity(enc, a, c, tw.device)
        kc.to_parquet(PL.serve_keysim_path(cfg, i), index=False)
        n_keys_total += len(kc)
        log(f"  batch {i:3d}: qids [{lo}, {hi}) -> {len(kc):,} candidate keys")
        del kc
        gc.collect()
    info.update({"serving_week": str(week.start), "cutoff": str(week.cutoff),
                 "n_live_articles": int(len(uni)), "n_keys": int(len(q)),
                 "n_candidate_keys": int(n_keys_total), "n_batches": len(batches(cfg, q)),
                 "seconds": round(time.time() - t0, 1)})
    paths["meta"].write_text(json.dumps(info, indent=2, default=str))
    log(f"  towers done: best epoch {info['best_epoch']} holdout R@12 "
        f"{info['best_holdout_recall@12']:.4f} ({info['seconds']:.0f}s)")
    return info


# ------------------------------------------------------------------ stage 2: rank and write


def training_frames(con, cfg, splits, log=print) -> tuple[list[pd.DataFrame], list[str]]:
    """Point-in-time matrices for the last ``train_weeks`` label weeks of the dataset."""
    from ensemble.completion.backtest import build_week, training_matrix

    n = int(cfg.track_b.train_weeks)
    weeks = [splits.test.shift(-k) for k in range(n - 1, -1, -1)]
    frames = []
    for week in weeks:
        art = build_week(con, cfg, week, log=log)
        frames.append(training_matrix(con, cfg, art, log=log))
        del art
        gc.collect()
    return frames, [str(w.start) for w in weeks]


def evidence_rows(df: pd.DataFrame, score: np.ndarray, chosen: dict[int, np.ndarray],
                  q: pd.DataFrame) -> pd.DataFrame:
    """The provenance a reason chip is allowed to cite, for the rows actually shown.

    Only columns that came out of the feature matrix are used, so an explanation
    can never assert co-purchase support for a pair that has none (D-035).
    """
    want = pd.DataFrame([(qid, int(a), r + 1) for qid, arr in chosen.items()
                         for r, a in enumerate(arr)], columns=["qid", "article_id", "rank"])
    cols = ["qid", "article_id", "a_co", "a_lift", "a_npmi", "s_co", "s_lift", "s_npmi",
            "backoff_level", "src_two_tower", "src_two_tower_content", "src_slot_pop",
            "tt_pair_sim", "ttc_pair_sim", "clip_sim", "same_colour_master", "price_tier_diff"]
    ev = df[[c for c in cols if c in df.columns]].copy()
    ev["score"] = score
    out = want.merge(ev, on=["qid", "article_id"], how="left")
    return out.merge(q[["qid", "anchor", "target_slot"]], on="qid", how="left")


def source_of(row) -> str:
    """One provenance label per row, strongest evidence first."""
    if row.a_co == row.a_co and row.a_co >= 1:
        return "co_purchase"
    if row.s_co == row.s_co and row.s_co >= 1:
        return "style_co_purchase"
    if row.src_two_tower == 1 or row.src_two_tower_content == 1:
        return "visual_compatibility"
    if row.src_slot_pop == 1:
        return "popular_in_slot"
    return "other"


def build(log=print) -> dict:
    """Score every (anchor, target slot) with the shipped ranker and write the serving table."""
    import lightgbm  # noqa: F401  - fail here, not after an hour of scoring
    from ensemble.completion import ranker as R

    cfg = load_config()
    con = connect(cfg, read_only=True)
    t0 = time.time()
    paths = PL.serve_paths(cfg)
    if not all(p.exists() for p in paths.values()):
        log("  building serving tower artifacts in a subprocess (PyTorch / LightGBM OpenMP clash)")
        subprocess.run([sys.executable, "-m", "ensemble.completion.serve_round3", "towers"],
                       check=True, cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    from ensemble.data.splits import load_splits
    splits = load_splits(con, cfg)

    frames, train_weeks = training_frames(con, cfg, splits, log=log)
    all_features = FE.feature_names(frames[0])
    FE.check_groups(all_features)
    feats = FE.subset(all_features, ("personalization",))
    t = time.time()
    booster, info = R.fit(cfg, frames, feats, log=log)
    log(f"  {SHIPPED_MODEL}: trained on {train_weeks}, {info['train_rows']:,} rows, "
        f"{len(feats)} features, {booster.current_iteration()} trees ({time.time() - t:.0f}s)")
    del frames
    gc.collect()

    splits, week, uni, q = _prepare(con, cfg)
    tt, ttc = pd.read_parquet(paths["tt"]), pd.read_parquet(paths["ttc"])
    C.build_sources(con, cfg, tt, ttc)
    del tt, ttc
    gc.collect()
    FE.build_context(con, week, cfg, uni, q, None)

    per_slot = int(cfg.serving.ctl_per_slot)
    fill_back = bool(cfg.serving.get("diversity_fill_back", True))
    pool = max(per_slot, int((cfg.track_b.get("serving_rules") or {}).get("pool", 48)))
    attrs = con.execute("SELECT article_id, product_type_no, product_code FROM articles").df()
    ptype = dict(zip(attrs.article_id.to_numpy(), attrs.product_type_no.to_numpy()))
    pcode = dict(zip(attrs.article_id.to_numpy(), attrs.product_code.to_numpy()))
    parts: list[pd.DataFrame] = []
    n_rows = 0
    step = int(cfg.track_b.eval_chunk_queries)
    for i, (lo, hi) in enumerate(batches(cfg, q)):
        C.register_keys(con, lo, hi)
        FE.register_key_sims(con, pd.read_parquet(PL.serve_keysim_path(cfg, i)))
        n_batch, n_shown = 0, 0
        for sub in range(lo, hi, step):
            df = FE.chunk_features(con, cfg, sub, min(sub + step, hi), with_labels=False,
                                   personalize=False)
            if not len(df):
                continue
            score = booster.predict(df[feats], num_threads=8)
            top = R.top_k(df, score, pool)
            # The caps re-order and then back-fill to `per_slot` (D-035): applied as a hard
            # filter they leave modules part-empty and cost five times as much accuracy.
            chosen = {qid: FU.apply_diversity(arr, per_slot, ptype, pcode,
                                              max_per_product_type=int(cfg.serving.max_per_product_type),
                                              one_per_product_code=True, fill_back=fill_back)
                      for qid, arr in top.items()}
            parts.append(evidence_rows(df, score, chosen, q))
            n_batch += len(df)
            n_shown += sum(len(v) for v in chosen.values())
            del df, score, top, chosen
            gc.collect()
        n_rows += n_batch
        log(f"  batch {i:3d}: qids [{lo}, {hi}) {n_batch:,} candidate rows -> {n_shown:,} shown")
    ctl = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    ctl["source"] = [source_of(r) for r in ctl.itertuples(index=False)]
    ctl = ctl.rename(columns={"target_slot": "slot", "a_lift": "lift"})
    ctl = ctl[["anchor", "slot", "rank", "article_id", "source", "lift", "score", "a_co", "a_npmi",
               "s_co", "s_npmi", "s_lift", "backoff_level", "src_two_tower",
               "src_two_tower_content", "src_slot_pop", "tt_pair_sim", "ttc_pair_sim", "clip_sim",
               "same_colour_master", "price_tier_diff"]]
    out = cfg.path("processed") / "models" / "complete_the_look.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = out.with_name("complete_the_look_round1.parquet")
    if out.exists() and not keep.exists():
        # The Round-1 table is a generated artifact, but `make track-b serve` takes an hour
        # to rebuild it; keep one copy so the two serving tables can be compared.
        shutil.copy2(out, keep)
        log(f"  kept the previous serving table as {keep.name}")
    ctl.to_parquet(out, index=False)
    summary = {"model": SHIPPED_MODEL, "n_features": len(feats), "trees": booster.current_iteration(),
               "train_weeks": train_weeks, "serving_week": str(week.start), "cutoff": str(week.cutoff),
               "n_live_articles": int(len(uni)), "n_keys": int(len(q)), "n_scored_rows": int(n_rows),
               "n_rows": int(len(ctl)), "n_anchors": int(ctl.anchor.nunique()),
               "per_slot": per_slot, "pool": pool,
               "max_per_product_type": int(cfg.serving.max_per_product_type),
               "one_per_product_code": True, "diversity_fill_back": fill_back,
               "source_mix": {k: int(v) for k, v in ctl.source.value_counts().items()},
               "mean_distinct_product_types_per_module": float(
                   ctl.assign(t=ctl.article_id.map(ptype)).groupby(["anchor", "slot"])["t"].nunique().mean()),
               "seconds": round(time.time() - t0, 1), "fit": info,
               "top_features": [{"feature": f, "gain_share": round(g, 5)}
                                for f, g in R.importance(booster, feats)[:25]],
               "manifest": P.manifest(cfg, con, {"stage": "serve_round3"})}
    d = cfg.path("reports") / "track_b_round3"
    d.mkdir(parents=True, exist_ok=True)
    (d / "serving.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"wrote {out} ({len(ctl):,} rows, {ctl.anchor.nunique():,} anchors, "
        f"{summary['seconds']:.0f}s); provenance {summary['source_mix']}")
    return summary


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "build"
    log = lambda s: print(s, flush=True)  # noqa: E731
    if stage == "towers":
        build_towers(log=log)
    elif stage == "build":
        build(log=log)
    else:
        raise SystemExit("usage: python -m ensemble.completion.serve_round3 [towers|build]")


if __name__ == "__main__":
    main()

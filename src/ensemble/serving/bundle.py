"""Offline build of a versioned Track B serving bundle (D-041, D-042).

  python -m ensemble.serving.bundle build      # ~1 h; needs the ingested H&M data
  python -m ensemble.serving.bundle verify <bundle dir>

Serving week = the week after the data ends (cutoff 2020-09-22), as in
``ensemble.completion.serve_round3``: every mined pair, catalogue statistic and customer
aggregate is legal, nothing is held back. Steps:

1. train ``lgbm_compatibility`` (74 features) and ``lgbm_personalized`` (95 features) on the
   last two label weeks, each with its own point-in-time features (cached matrices);
2. score every (live anchor, other slot) key with the compatibility ranker and keep its top-P
   candidates (P from D-041) with their compatibility features, exact association NPMI /
   co-count and two-tower similarity, and the compatibility score;
3. snapshot every customer's point-in-time profile (104 weeks before the cutoff) as one
   compressed row per customer in SQLite;
4. export the personalized ranker to the NumPy runtime (``serving.gbdt``);
5. **equivalence gate**: for sampled real (customer, anchor, slot) requests, score the pool
   offline (feature SQL + LightGBM) and online (profile + NumPy runtime); refuse to publish
   if any ordering or score disagrees beyond tolerance;
6. write the catalogue / availability snapshot, slot popularity, the FashionCLIP visual
   index (shop-side adapter in a PyTorch subprocess), a manifest with sha256 per file, then
   publish atomically: build in ``.tmp-*``, rename, then swap the ``CURRENT`` pointer.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble.config import ROOT, load_config
from ensemble.serving.runtime import (AUX, BUNDLE_FORMAT, PERSONAL, SLOTS, encode_profile, personal_matrix,
                                      sha256_file)


def bundles_root(cfg) -> Path:
    d = cfg.path("processed") / "serving_bundles"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Profiles (point-in-time customer aggregates, mirrors features._customer_tables)
# ---------------------------------------------------------------------------

PROFILE_SQL = """
WITH h AS (
    SELECT t.customer_idx, t.article_id, t.t_dat, t.price,
           date_diff('day', t.t_dat, {start}) AS days_ago,
           1.0 / (1 + date_diff('day', t.t_dat, {start}) / 7.0) AS w,
           a.slot, a.product_code, a.product_type_no, a.department_no, a.section_no, a.garment_group_no,
           TRY_CAST(a.colour_group_code AS INTEGER) AS colour_group_code, a.perceived_colour_master_id
    FROM transactions t JOIN articles a USING (article_id)
    WHERE t.t_dat >= {lo} AND t.t_dat < {start} {cust_filter}),
cust AS (SELECT customer_idx, count(*) AS n, count(DISTINCT t_dat) AS d, min(days_ago) AS l, avg(price) AS p,
                sum(w) AS w FROM h GROUP BY 1),
{dims},
art AS (SELECT customer_idx, list(struct_pack(k := article_id, n := n, d := d) ORDER BY article_id) AS art FROM (
          SELECT customer_idx, article_id, count(*) AS n, min(days_ago) AS d FROM h GROUP BY 1, 2) GROUP BY 1),
style AS (SELECT customer_idx, list(struct_pack(k := product_code, n := n) ORDER BY product_code) AS style FROM (
          SELECT customer_idx, product_code, count(*) AS n FROM h GROUP BY 1, 2) GROUP BY 1)
SELECT c.customer_idx, to_json(struct_pack(n := c.n, d := c.d, l := c.l, p := c.p, w := c.w,
       {dim_fields}, art := art.art, style := style.style)) AS j
FROM cust c {dim_joins} LEFT JOIN art USING (customer_idx) LEFT JOIN style USING (customer_idx)
ORDER BY c.customer_idx
"""
DIM_COLS = {"type": "product_type_no", "dept": "department_no", "section": "section_no",
            "ggroup": "garment_group_no", "colour": "colour_group_code", "cmaster": "perceived_colour_master_id",
            "slot": "slot"}


def profile_sql(week_start, weeks: int, customers_table: str | None = None) -> str:
    start = f"DATE '{week_start}'"
    lo = f"DATE '{week_start - timedelta(weeks=int(weeks))}'"
    dims = ",\n".join(
        f"""d_{d} AS (SELECT customer_idx, list(struct_pack(k := key, n := n, w := w) ORDER BY key) AS {d} FROM (
              SELECT customer_idx, {c} AS key, count(*) AS n, sum(w) AS w FROM h WHERE {c} IS NOT NULL
              GROUP BY 1, 2) GROUP BY 1)""" for d, c in DIM_COLS.items())
    return PROFILE_SQL.format(
        start=start, lo=lo, dims=dims,
        dim_fields=", ".join(f"{d} := d_{d}.{d}" for d in DIM_COLS),
        dim_joins=" ".join(f"LEFT JOIN d_{d} USING (customer_idx)" for d in DIM_COLS),
        cust_filter=f"AND t.customer_idx IN (SELECT customer_idx FROM {customers_table})" if customers_table else "")


def write_profiles(con, path: Path, week_start, weeks: int, log=print) -> dict:
    t0 = time.time()
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE profiles (customer_idx INTEGER PRIMARY KEY, blob BLOB NOT NULL)")
    n, nbytes = 0, 0
    cur = con.execute(profile_sql(week_start, weeks))
    while True:
        rows = cur.fetchmany(50_000)
        if not rows:
            break
        batch = []
        for cid, j in rows:
            blob = encode_profile(json.loads(j))
            nbytes += len(blob)
            batch.append((int(cid), blob))
        db.executemany("INSERT INTO profiles VALUES (?, ?)", batch)
        n += len(batch)
    db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    db.execute("INSERT INTO meta VALUES ('snapshot_week_start', ?), ('history_weeks', ?)", (str(week_start), str(weeks)))
    db.commit()
    db.execute("VACUUM")
    db.close()
    log(f"  profiles: {n:,} customers, {nbytes / 1e6:.0f} MB compressed blobs ({time.time() - t0:.0f}s)")
    return {"n_customers": n, "blob_bytes": nbytes, "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _pool_rows(df: pd.DataFrame, score: np.ndarray, pool: int) -> np.ndarray:
    """Row positions of each qid's top-``pool`` by score (ties -> smaller article id), in rank order."""
    qid, art = df.qid.to_numpy(), df.article_id.to_numpy()
    order = np.lexsort((art, -score, qid))
    q_s = qid[order]
    starts = np.r_[0, np.flatnonzero(q_s[1:] != q_s[:-1]) + 1]
    pos = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    return order[pos < pool]


def build(log=print, pool: int | None = None, n_equivalence: int = 400) -> Path:
    import lightgbm as lgb

    from ensemble.completion import candidates as C
    from ensemble.completion import features as FE
    from ensemble.completion import pipeline as PL
    from ensemble.completion import protocol as CP
    from ensemble.completion import ranker as R
    from ensemble.completion.serve_round3 import _prepare, batches, training_frames
    from ensemble.data.splits import load_splits
    from ensemble.db import connect
    from ensemble.research.protocol import data_fingerprint
    from ensemble.serving.gbdt import Forest

    cfg = load_config()
    sb = cfg.serving.get("bundle") or {}
    pool = int(pool or sb.get("pool", 48))
    con = connect(cfg, read_only=True)
    t0 = time.time()
    paths = PL.serve_paths(cfg)
    if not all(p.exists() for p in paths.values()):
        log("  serving-week tower artifacts missing: building them in a PyTorch subprocess")
        subprocess.run([sys.executable, "-m", "ensemble.completion.serve_round3", "towers"], check=True,
                       cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    splits = load_splits(con, cfg)
    root = bundles_root(cfg)
    tmp = root / f".tmp-{int(time.time())}"
    tmp.mkdir()
    try:
        # 1. rankers
        frames, train_weeks = training_frames(con, cfg, splits, log=log)
        feats_all = FE.feature_names(frames[0])
        FE.check_groups(feats_all)
        feats_comp = FE.subset(feats_all, ("personalization",))
        t = time.time()
        b_comp, info_comp = R.fit(cfg, frames, feats_comp, log=log)
        b_pers, info_pers = R.fit(cfg, frames, feats_all, log=log)
        fit_s = time.time() - t
        del frames
        gc.collect()
        pers_names = list(b_pers.feature_name())
        assert pers_names == feats_all and sorted(set(pers_names) - set(feats_comp)) == sorted(PERSONAL), \
            "personalization feature set differs from the serving runtime's PERSONAL list"
        pool_feats = [f for f in pers_names if f not in PERSONAL]
        assert set(pool_feats) == set(feats_comp)
        b_comp.save_model(str(tmp / "model_compatibility.txt"))
        b_pers.save_model(str(tmp / "model_personalized.txt"))
        forest = Forest.from_booster_dump(b_pers.dump_model())
        model_sha = forest.save(tmp / "model_personalized.json")
        log(f"  rankers: compatibility {b_comp.current_iteration()} trees / {len(feats_comp)} features, "
            f"personalized {b_pers.current_iteration()} trees / {len(feats_all)} features ({fit_s:.0f}s)")

        # 2. pools over the serving key space
        splits, week, uni, q = _prepare(con, cfg)
        tt, ttc = pd.read_parquet(paths["tt"]), pd.read_parquet(paths["ttc"])
        C.build_sources(con, cfg, tt, ttc)
        del tt, ttc
        FE.build_context(con, week, cfg, uni, q, None)
        q_key = q.set_index("qid")[["anchor", "target_slot"]]
        parts_dir = tmp / "_parts"
        parts_dir.mkdir()
        n_parts, sizes, keys, arts_l, comp_l = 0, [], [], [], []
        step = int(cfg.track_b.eval_chunk_queries)
        t = time.time()
        for i, (lo, hi) in enumerate(batches(cfg, q)):
            C.register_keys(con, lo, hi)
            FE.register_key_sims(con, pd.read_parquet(PL.serve_keysim_path(cfg, i)))
            for sub in range(lo, hi, step):
                df = FE.chunk_features(con, cfg, sub, min(sub + step, hi), with_labels=False, personalize=False)
                if not len(df):
                    continue
                score = b_comp.predict(df[feats_comp], num_threads=8)
                rows = _pool_rows(df, score, pool)
                sel = df.iloc[rows].reset_index(drop=True)
                np.save(parts_dir / f"f{n_parts:05d}.npy", sel[pool_feats].to_numpy(np.float32))
                np.save(parts_dir / f"x{n_parts:05d}.npy", _exact_aux(con, sel, q_key))
                arts_l.append(sel.article_id.to_numpy(np.int64))
                comp_l.append(score[rows].astype(np.float64))
                keys.append(sel.qid.to_numpy())
                sizes.append(len(sel))
                n_parts += 1
                del df, score, sel
                gc.collect()
            log(f"  pools batch {i}: qids [{lo}, {hi}) ({time.time() - t:.0f}s)")
        qids = np.concatenate(keys)
        art = np.concatenate(arts_l)
        comp = np.concatenate(comp_l)
        # Concatenate the on-disk parts into memory-mapped arrays: never more than one part resident.
        feat = np.lib.format.open_memmap(tmp / "pool_feat.npy", mode="w+", dtype=np.float32,
                                         shape=(len(art), len(pool_feats)))
        aux = np.lib.format.open_memmap(tmp / "pool_aux.npy", mode="w+", dtype=np.float64, shape=(len(art), len(AUX)))
        o = 0
        for j, n_ in enumerate(sizes):
            feat[o:o + n_] = np.load(parts_dir / f"f{j:05d}.npy")
            aux[o:o + n_] = np.load(parts_dir / f"x{j:05d}.npy")
            o += n_
        feat.flush()
        aux.flush()
        feat_gb = feat.nbytes / 1e9
        del feat, aux
        shutil.rmtree(parts_dir)
        gc.collect()
        kq = q_key.loc[qids]
        anchor = kq.anchor.to_numpy(np.int64)
        slot_id = np.array([SLOTS.index(s) for s in kq.target_slot], dtype=np.int8)
        # Rows are grouped by qid and qids are sorted by (anchor, target_slot): a key is a contiguous run.
        change = np.r_[True, (anchor[1:] != anchor[:-1]) | (slot_id[1:] != slot_id[:-1])]
        starts = np.flatnonzero(change)
        counts = np.diff(np.r_[starts, len(anchor)])
        order_check = np.lexsort((slot_id[starts], anchor[starts]))
        assert (order_check == np.arange(len(starts))).all(), "pool keys are not sorted by (anchor, slot)"
        np.savez(tmp / "keys.npz", anchor=anchor[starts], slot_id=slot_id[starts], offset=starts.astype(np.int64),
                 count=counts.astype(np.int32))
        np.save(tmp / "pool_article.npy", art)
        np.save(tmp / "pool_compat.npy", comp)
        log(f"  pools: {len(starts):,} keys, {len(art):,} rows, {feat_gb:.2f} GB features")

        # 3. catalogue + availability snapshot + slot popularity
        cat = con.execute(f"""
            WITH p AS (SELECT article_id, avg(price) AS cp_mean_price FROM transactions
                       WHERE t_dat BETWEEN DATE '{week.start - timedelta(weeks=52)}' AND DATE '{week.cutoff}' GROUP BY 1)
            SELECT a.article_id, a.product_code, a.product_type_no, a.product_type_name, a.department_no,
                   a.section_no, a.garment_group_no, TRY_CAST(a.colour_group_code AS INTEGER) AS colour_group_code,
                   a.perceived_colour_master_id, a.slot, a.is_jewellery, a.prod_name, a.colour_group_name,
                   a.index_group_name, p.cp_mean_price
            FROM articles a LEFT JOIN p USING (article_id) ORDER BY a.article_id""").df()
        u = uni.set_index("article_id")
        cat["live"] = cat.article_id.isin(u.index)
        cat["pop_recent"] = cat.article_id.map(u.pop_recent).fillna(0).astype(np.int64)
        cat["pop_window"] = cat.article_id.map(u.pop_window).fillna(0).astype(np.int64)
        cat.to_parquet(tmp / "catalog.parquet", index=False)
        sp = {s: g.sort_values(["pop_recent", "pop_window", "article_id"], ascending=[False, False, True])
              .article_id.head(200).astype(int).tolist() for s, g in uni.groupby("slot")}
        (tmp / "slot_popularity.json").write_text(json.dumps(sp))

        # 4. profiles
        prof = write_profiles(con, tmp / "profiles.sqlite", week.start, int(cfg.track_b.customer_weeks), log=log)

        # 5. schema + equivalence gate
        schema = {"personalized_features": pers_names, "pool_features": pool_feats, "personal_features": PERSONAL,
                  "aux": AUX, "categorical": [c for c in FE.CATEGORICAL if c in pers_names]}
        (tmp / "feature_schema.json").write_text(json.dumps(schema, indent=1))
        eq = equivalence_gate(con, cfg, week, uni, q, tmp, b_pers, pers_names, pool_feats, n_equivalence, log)
        (tmp / "equivalence.json").write_text(json.dumps(eq, indent=1, default=float))
        if not eq["passed"]:
            raise RuntimeError(f"equivalence gate failed: {eq['summary']}")

        # 6. visual index (PyTorch subprocess for the shop-side adapter)
        vis = _visual_index(cfg, tmp, cat, log)

        catalog_version = hashlib.sha1((str(week.start) + ",".join(map(str, sorted(uni.article_id.astype(int)))))
                                       .encode()).hexdigest()[:10]
        profile_version = f"{week.start}-{prof['n_customers']}"
        model_version = model_sha[:10]
        version = f"{week.start}_{model_version}_{catalog_version}"
        files = {}
        for p in sorted(tmp.rglob("*")):
            if p.is_file():
                files[str(p.relative_to(tmp))] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
        manifest = {
            "format": BUNDLE_FORMAT, "bundle_version": version, "model_version": model_version,
            "catalog_version": catalog_version, "profile_version": profile_version,
            "feature_schema_sha1": hashlib.sha1(json.dumps(schema, sort_keys=True).encode()).hexdigest()[:12],
            "serving_week": str(week.start), "cutoff": str(week.cutoff), "train_weeks": train_weeks,
            "pool": pool, "n_keys": int(len(starts)), "n_pool_rows": int(len(art)),
            "n_live_articles": int(len(uni)), "profiles": prof, "visual": vis,
            "rankers": {"compatibility": {"trees": b_comp.current_iteration(), "features": len(feats_comp),
                                          "fit": info_comp},
                        "personalized": {"trees": b_pers.current_iteration(), "features": len(feats_all),
                                         "fit": info_pers}},
            "equivalence": eq["summary"], "files": files, "lightgbm": lgb.__version__,
            "data_fingerprint": data_fingerprint(con),
            "manifest": CP.manifest(cfg, con, {"stage": "serving_bundle"}),
            "build_seconds": round(time.time() - t0, 1)}
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
        final = root / version
        if final.exists():
            shutil.rmtree(final)
        tmp.rename(final)
        publish(root, version)
        log(f"published bundle {version} ({manifest['build_seconds']:.0f}s)")
        return final
    except BaseException:
        log(f"  build failed; leaving {tmp.name} for inspection (CURRENT unchanged)")
        raise


def publish(root: Path, version: str) -> None:
    """Atomically point ``CURRENT`` at ``version`` (write a temp file, then rename over)."""
    if not (root / version / "manifest.json").exists():
        raise FileNotFoundError(f"bundle {version} not found under {root}")
    tmp = root / ".CURRENT.tmp"
    tmp.write_text(version)
    os.replace(tmp, root / "CURRENT")


def _exact_aux(con, sel: pd.DataFrame, q_key: pd.DataFrame) -> np.ndarray:
    """Exact (float64) association NPMI and co-count and two-tower similarity for pool rows: the
    personalization interactions multiply these before the float32 cast, as the SQL does."""
    k = q_key.loc[sel.qid.to_numpy()]
    df = pd.DataFrame({"i": np.arange(len(sel)), "anchor": k.anchor.to_numpy(np.int64),
                       "target_slot": k.target_slot.to_numpy(), "article_id": sel.article_id.to_numpy(np.int64)})
    con.register("_bx_df", df)
    r = con.execute("""SELECT x.i, a.npmi, a.co, ks.tt_pair_sim FROM _bx_df x
                       LEFT JOIN _tb_assoc a ON a.src = x.anchor AND a.dst_slot = x.target_slot AND a.dst = x.article_id
                       LEFT JOIN _tb_keysim ks ON ks.anchor = x.anchor AND ks.target_slot = x.target_slot
                                               AND ks.article_id = x.article_id
                       ORDER BY x.i""").df()
    con.unregister("_bx_df")
    assert len(r) == len(sel), "duplicate association rows for one pair"
    return r[["npmi", "co", "tt_pair_sim"]].to_numpy(np.float64)


def equivalence_gate(con, cfg, week, uni, q, bdir: Path, booster, pers_names, pool_feats, n: int, log) -> dict:
    """Offline (SQL features + LightGBM) vs online (profile + NumPy runtime) on sampled requests."""
    from ensemble.completion import candidates as C
    from ensemble.completion import features as FE
    from ensemble.completion import pipeline as PL
    from ensemble.completion.serve_round3 import batches
    from ensemble.serving.runtime import decode_profile
    rng = np.random.default_rng(7)
    keys = np.load(bdir / "keys.npz")
    b_batches = batches(cfg, q)
    first_hi = b_batches[0][1]
    anchors0 = set(q.anchor[q.qid < first_hi].astype(int))
    key_rows = [i for i, a in enumerate(keys["anchor"]) if int(a) in anchors0]
    pick = rng.choice(key_rows, size=min(n, len(key_rows)), replace=False)
    prof_db = sqlite3.connect(bdir / "profiles.sqlite")
    cust_ids = np.array([r[0] for r in prof_db.execute(
        "SELECT customer_idx FROM profiles WHERE customer_idx % 97 = 3 LIMIT 5000")])
    prof_db.close()
    custs = rng.choice(cust_ids, size=len(pick))
    custs[::5] = 1_999_999_999           # every fifth request: a customer with no profile
    eq_q = pd.DataFrame({"customer_idx": custs.astype(np.int64), "anchor": keys["anchor"][pick].astype(np.int64),
                         "target_slot": [SLOTS[s] for s in keys["slot_id"][pick]]})
    eq_q["anchor_slot"] = eq_q.anchor.map(uni.set_index("article_id").slot)
    eq_q["t_dat"] = pd.NaT
    eq_q["truth"] = [np.empty(0, dtype=np.int64)] * len(eq_q)
    eq_q = eq_q.rename_axis("qid").reset_index()
    C.register(con, eq_q, uni)
    FE.build_context(con, week, cfg, uni, eq_q, pd.read_parquet(PL.serve_keysim_path(cfg, 0)))
    off = FE.chunk_features(con, cfg, 0, len(eq_q), with_labels=False, personalize=True)
    off_score = booster.predict(off[pers_names], num_threads=4)
    pool_art = np.load(bdir / "pool_article.npy", mmap_mode="r")
    pool_feat = np.load(bdir / "pool_feat.npy", mmap_mode="r")
    pool_aux = np.load(bdir / "pool_aux.npy", mmap_mode="r")
    cat = pd.read_parquet(bdir / "catalog.parquet").set_index("article_id", drop=False)
    from ensemble.serving.gbdt import Forest
    forest = Forest.load(bdir / "model_personalized.json")
    pcols = [pers_names.index(f) for f in pool_feats]
    xcols = [pers_names.index(f) for f in PERSONAL]
    prof_db = sqlite3.connect(bdir / "profiles.sqlite")
    agree, max_diff, max_feat_diff, n_req = 0, 0.0, 0.0, 0
    rows = []
    for qi, (_, r) in enumerate(eq_q.iterrows()):
        o, c = int(keys["offset"][pick[qi]]), int(keys["count"][pick[qi]])
        arts = np.asarray(pool_art[o:o + c], dtype=np.int64)
        # offline: the same pool's rows from the SQL feature matrix, scored by LightGBM
        m = (off.qid == r.qid).to_numpy()
        oq, sq = off[m], off_score[m]
        idx = np.searchsorted(oq.article_id.to_numpy(), arts)
        assert (oq.article_id.to_numpy()[np.minimum(idx, len(oq) - 1)] == arts).all(), "pool row missing offline"
        X_off, s_off = oq[pers_names].to_numpy(np.float32)[idx], sq[idx]
        # online
        blob = prof_db.execute("SELECT blob FROM profiles WHERE customer_idx = ?", (int(r.customer_idx),)).fetchone()
        prof = decode_profile(blob[0]) if blob else None
        cc = cat.loc[arts]
        cand = {k_: cc[k_].to_numpy() for k_ in ("product_code", "product_type_no", "department_no", "section_no",
                                                  "garment_group_no", "colour_group_code",
                                                  "perceived_colour_master_id", "cp_mean_price")}
        cand["article_id"] = arts
        P = personal_matrix(prof, cand, r.target_slot, np.asarray(pool_aux[o:o + c]))
        X = np.empty((c, len(pers_names)), dtype=np.float32)
        X[:, pcols] = np.asarray(pool_feat[o:o + c])
        X[:, xcols] = P
        s_on = forest.predict(X)
        fd = np.max(np.abs(np.nan_to_num(X, nan=-7.0) - np.nan_to_num(X_off, nan=-7.0)))
        same = (np.lexsort((arts, -s_on)) == np.lexsort((arts, -s_off))).all()
        agree += int(same)
        max_diff = max(max_diff, float(np.max(np.abs(s_on - s_off))))
        max_feat_diff = max(max_feat_diff, float(fd))
        n_req += 1
        rows.append({"customer_has_profile": prof is not None, "pool": c, "same_order": bool(same),
                     "max_score_diff": float(np.max(np.abs(s_on - s_off))), "max_feature_diff": float(fd)})
    prof_db.close()
    del bundle
    summary = {"requests": n_req, "identical_order": agree, "identical_order_share": agree / max(n_req, 1),
               "max_score_abs_diff": max_diff, "max_feature_abs_diff": max_feat_diff,
               "with_profile": int(sum(r_["customer_has_profile"] for r_ in rows)),
               "tolerance": {"identical_order_share": 1.0, "max_score_abs_diff": 1e-6}}
    passed = summary["identical_order_share"] >= 1.0 and max_diff <= 1e-6
    log(f"  equivalence gate: {agree}/{n_req} identical orderings, max |score diff| {max_diff:.2e}, "
        f"max |feature diff| {max_feat_diff:.2e} -> {'PASS' if passed else 'FAIL'}")
    return {"passed": passed, "summary": summary, "requests": rows}


def _visual_index(cfg, bdir: Path, cat: pd.DataFrame, log) -> dict:
    d = cfg.path("processed") / "clip"
    if not (d / "article_emb.npy").exists():
        log("  visual index: no FashionCLIP embeddings; bundle ships without one (visual paths degrade)")
        return {"present": False}
    ids, emb, has = np.load(d / "article_ids.npy"), np.load(d / "article_emb.npy"), np.load(d / "has_image.npy")
    pos = pd.Series(np.arange(len(ids)), index=ids)
    live = cat.article_id[cat.live & cat.article_id.isin(ids)].to_numpy(np.int64)
    live = live[has[pos[live].to_numpy()]]
    v = emb[pos[live].to_numpy()].astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    out = bdir / "visual"
    out.mkdir()
    np.save(out / "live_ids.npy", live)
    np.save(out / "clip.npy", v.astype(np.float16))
    all_ids = ids[has].astype(np.int64)
    order = np.argsort(all_ids)
    np.save(out / "all_ids.npy", all_ids[order])
    np.save(out / "all_clip.npy", emb[has][order].astype(np.float16))
    r = subprocess.run([sys.executable, "-m", "ensemble.serving.visual_index", str(out)], cwd=str(ROOT),
                       env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
    adapted = r.returncode == 0 and (out / "clip_shop_adapted.npy").exists()
    if not adapted:
        log(f"  visual index: shop-side adapter unavailable ({r.stderr.strip()[-300:]}); raw FashionCLIP only")
    log(f"  visual index: {len(live):,} live articles with images (adapter {'yes' if adapted else 'no'})")
    return {"present": True, "n": int(len(live)), "adapted": bool(adapted)}


def verify(path: str) -> dict:
    from ensemble.serving.runtime import Bundle
    b = Bundle(Path(path), verify_hashes=True)
    return {"bundle": b.version, "ok": True, "load_seconds": b.load_seconds}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build(log=lambda s: print(s, flush=True))
    elif cmd == "verify":
        print(json.dumps(verify(sys.argv[2])))
    elif cmd == "publish":
        publish(bundles_root(load_config()), sys.argv[2])
    else:
        raise SystemExit(__doc__)

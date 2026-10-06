"""A tiny synthetic serving bundle: exercises every online path without the restricted H&M data.

  python -m ensemble.serving.fixture <out_dir>      # also used by tests/test_serving.py and `make serve-smoke`

Catalogue: 48 articles in four slots, two colourways per style; articles >= 140 are not live.
Pools exist for live upper and lower anchors; ``full`` / ``socks`` / ``swimwear`` have no pool
and no popular items (the empty-slot path). Customers 1 and 2 have profiles, others do not.
The personalized model is a hand-written three-tree forest whose second tree splits on
``cu_type_share``, so personalization visibly re-orders a pool.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble.serving.runtime import AUX, BUNDLE_FORMAT, PERSONAL, SLOTS, encode_profile, sha256_file

POOL_FEATURES = ["a_co", "a_lift", "a_npmi", "s_co", "s_lift", "backoff_level", "src_two_tower",
                 "src_two_tower_content", "src_slot_pop", "tt_pair_sim", "clip_sim", "same_colour_master",
                 "price_tier_diff", "cand_pop_recent"]
FIX_SLOTS = ["upper", "lower", "shoes", "accessories"]


def _tree(feature: str, threshold: float, left: float, right: float, names: list[str], missing="None",
          default_left=False) -> dict:
    return {"feature": [names.index(feature)], "threshold": [threshold], "is_cat": [False],
            "default_left": [default_left], "missing": [{"None": 0, "Zero": 1, "NaN": 2}[missing]],
            "left": [-1], "right": [-2], "cat_sets": [[]], "leaf_value": [left, right], "root": 0}


def build(out: Path, with_visual: bool = True, version: str = "fixture-v1") -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    ids = np.arange(100, 148)
    slot = [FIX_SLOTS[(i // 2) % 4] for i in range(len(ids))]
    cat = pd.DataFrame({
        "article_id": ids, "product_code": 1000 + ids // 2, "product_type_no": [10 + (i % 6) for i in range(len(ids))],
        "product_type_name": [f"Type {10 + (i % 6)}" for i in range(len(ids))],
        "department_no": [200 + (i % 3) for i in range(len(ids))], "section_no": [30 + (i % 4) for i in range(len(ids))],
        "garment_group_no": [40 + (i % 2) for i in range(len(ids))],
        "colour_group_code": [i % 5 for i in range(len(ids))], "perceived_colour_master_id": [i % 3 for i in range(len(ids))],
        "slot": slot, "is_jewellery": [False] * len(ids), "prod_name": [f"Article {a}" for a in ids],
        "colour_group_name": ["Black"] * len(ids), "index_group_name": ["Ladieswear"] * len(ids),
        "cp_mean_price": np.round(0.01 + 0.001 * (ids % 7), 6),
        "live": ids < 140, "pop_recent": (200 - ids).astype(np.int64), "pop_window": (300 - ids).astype(np.int64)})
    cat.to_parquet(out / "catalog.parquet", index=False)
    live = cat[cat.live]
    sp = {s: g.sort_values(["pop_recent", "article_id"], ascending=[False, True]).article_id.astype(int).tolist()
          for s, g in live.groupby("slot")}
    (out / "slot_popularity.json").write_text(json.dumps(sp))
    # Pools: every live upper / lower anchor, for every other fixture slot.
    k_anchor, k_slot, k_off, k_cnt, arts, comp, feat, aux = [], [], [], [], [], [], [], []
    for a, s in zip(live.article_id, live.slot):
        if s not in ("upper", "lower"):
            continue
        for t in FIX_SLOTS:
            if t == s:
                continue
            cands = live.article_id[live.slot == t].to_numpy()
            k_anchor.append(int(a)); k_slot.append(SLOTS.index(t)); k_off.append(len(arts)); k_cnt.append(len(cands))
            for c in cands:
                co = float(rng.integers(0, 8)) if rng.random() < 0.5 else np.nan
                f = [co, (co * 3.0) if co == co else np.nan, (0.1 * co) if co == co else np.nan,
                     float(rng.integers(0, 4)) if rng.random() < 0.3 else np.nan, np.nan,
                     0.0 if co == co else 2.0, float(rng.random() < 0.5), float(rng.random() < 0.3),
                     1.0, float(rng.random()), float(rng.random()), float(rng.random() < 0.4),
                     float(rng.integers(-1, 2)), float(200 - c)]
                arts.append(int(c)); feat.append(f)
                comp.append((0.0 if co != co else co) * 0.1 + (200 - c) * 0.01)
                aux.append([(0.1 * co) if co == co else np.nan, co, f[9]])
    order = np.lexsort((k_slot, k_anchor))
    assert (order == np.arange(len(order))).all()
    np.savez(out / "keys.npz", anchor=np.array(k_anchor, np.int64), slot_id=np.array(k_slot, np.int8),
             offset=np.array(k_off, np.int64), count=np.array(k_cnt, np.int32))
    np.save(out / "pool_article.npy", np.array(arts, np.int64))
    np.save(out / "pool_compat.npy", np.array(comp, np.float64))
    np.save(out / "pool_feat.npy", np.array(feat, np.float32))
    np.save(out / "pool_aux.npy", np.array(aux, np.float64))
    names = POOL_FEATURES + PERSONAL
    forest = {"feature_names": names, "objective": "lambdarank", "num_trees": 3, "trees": [
        _tree("cand_pop_recent", 70.0, -0.5, 0.5, names),
        _tree("cu_type_share", 0.2, 0.0, 3.0, names, missing="NaN", default_left=True),
        _tree("a_co", 2.5, 0.0, 0.25, names, missing="NaN", default_left=True)]}
    body = json.dumps(forest, separators=(",", ":"))
    (out / "model_personalized.json").write_text(body)
    schema = {"personalized_features": names, "pool_features": POOL_FEATURES, "personal_features": PERSONAL,
              "aux": AUX, "categorical": []}
    (out / "feature_schema.json").write_text(json.dumps(schema, indent=1))
    db = sqlite3.connect(out / "profiles.sqlite")
    db.execute("CREATE TABLE profiles (customer_idx INTEGER PRIMARY KEY, blob BLOB NOT NULL)")
    profiles = {
        1: {"n": 6, "d": 3, "l": 5, "p": 0.02, "w": 3.0,
            "type": [{"k": 13, "n": 4, "w": 2.4}, {"k": 10, "n": 2, "w": 0.6}], "dept": [{"k": 200, "n": 6, "w": 3.0}],
            "section": [{"k": 30, "n": 6, "w": 3.0}], "ggroup": [{"k": 40, "n": 6, "w": 3.0}],
            "colour": [{"k": 1, "n": 6, "w": 3.0}], "cmaster": [{"k": 1, "n": 6, "w": 3.0}],
            "slot": [{"k": "shoes", "n": 6, "w": 3.0}], "art": [{"k": 131, "n": 1, "d": 5}],
            "style": [{"k": 1065, "n": 1}]},
        2: {"n": 2, "d": 1, "l": 40, "p": 0.015, "w": 0.4,
            "type": [{"k": 11, "n": 2, "w": 0.4}], "dept": [{"k": 201, "n": 2, "w": 0.4}], "section": [],
            "ggroup": [], "colour": [], "cmaster": [], "slot": [{"k": "accessories", "n": 2, "w": 0.4}],
            "art": [], "style": []}}
    db.executemany("INSERT INTO profiles VALUES (?, ?)", [(c, encode_profile(p)) for c, p in profiles.items()])
    db.commit()
    db.close()
    if with_visual:
        (out / "visual").mkdir(exist_ok=True)
        v = rng.normal(size=(len(ids), 512)).astype(np.float32)
        # Non-live upper article 140 (no live colourway) looks like live upper article 109.
        v[ids == 140] = v[ids == 109] + 0.05 * rng.normal(size=512)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        lv = cat.live.to_numpy()
        np.save(out / "visual" / "all_ids.npy", ids.astype(np.int64))
        np.save(out / "visual" / "all_clip.npy", v.astype(np.float16))
        np.save(out / "visual" / "live_ids.npy", ids[lv].astype(np.int64))
        np.save(out / "visual" / "clip.npy", v[lv].astype(np.float16))
    files = {str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
             for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    manifest = {"format": BUNDLE_FORMAT, "bundle_version": version, "model_version": hashlib.sha256(
        body.encode()).hexdigest()[:10], "catalog_version": "fixturecat", "profile_version": "fixture-2",
        "feature_schema_sha1": hashlib.sha1(json.dumps(schema, sort_keys=True).encode()).hexdigest()[:12],
        "serving_week": "2020-09-23", "cutoff": "2020-09-22", "pool": 24, "n_keys": len(k_anchor),
        "n_pool_rows": len(arts), "n_live_articles": int(cat.live.sum()), "profiles": {"n_customers": 2},
        "visual": {"present": with_visual, "adapted": False}, "files": files, "synthetic": True}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return out


if __name__ == "__main__":
    print(build(Path(sys.argv[1])))

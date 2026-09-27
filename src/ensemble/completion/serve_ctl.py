"""Precompute Complete the Look for every live anchor (serving input for M6).

  python -m ensemble.completion.serve_ctl

Runs in its own process because it uses PyTorch (see ``ensemble.api.build``).
Output: data/processed/models/complete_the_look.parquet
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd
import torch

from ensemble.completion import models as M
from ensemble.config import Config, load_config


def diversify(items: list[int], ptype: dict[int, int], style: dict[int, int], k: int, max_per_type: int) -> list[int]:
    """Re-rank for diversity: at most ``max_per_type`` per product type, one colourway per style."""
    out, count, styles = [], defaultdict(int), set()
    for a in items:
        if count[ptype.get(a)] < max_per_type and style.get(a) not in styles:
            out.append(a)
            count[ptype.get(a)] += 1
            styles.add(style.get(a))
        if len(out) == k:
            break
    return out


def complete_the_look(cfg) -> pd.DataFrame:
    models = cfg.path("processed") / "models"
    uni = pd.read_parquet(models / "completion_universe.parquet")
    pairs = pd.read_parquet(models / "completion_pairs.parquet")
    with open(models / "two_tower_encoder.pkl", "rb") as f:
        art = pickle.load(f)
    enc = art["enc"]
    model = M.TwoTower(enc.cardinality, Config(art["tt"]))
    model.load_state_dict(torch.load(models / "two_tower.pt", map_location="cpu"))
    w = json.loads((cfg.path("reports") / "m4_selected.json").read_text())["rrf_weight"]
    k, per = int(cfg.completion.k), int(cfg.serving.ctl_per_slot)

    anchors = uni[["article_id", "slot"]]
    q = pd.DataFrame([(a, s, t) for a, s in anchors.itertuples(index=False) for t in M.SLOT_IDS if t != s],
                     columns=["anchor", "anchor_slot", "target_slot"])
    tt = M.two_tower_recs(model, enc, q, uni, 4 * k, "cpu")
    pairs = pairs.sort_values(["src", "dst_slot", "npmi"], ascending=[True, True, False])
    live = set(uni.article_id)
    assoc = {key: g.dst.values[:4 * k] for key, g in pairs[pairs.dst.isin(live)].groupby(["src", "dst_slot"], sort=False)}
    lift = {(a, b): l for a, b, l in pairs[["src", "dst", "lift"]].itertuples(index=False)}
    ptype = dict(zip(uni.article_id, enc.codes[[enc.row[a] for a in uni.article_id], 0].tolist()))
    style = dict(duckdb.sql(f"SELECT CAST(article_id AS INTEGER), product_code FROM read_csv('{cfg.path('raw') / 'articles.csv'}', "
                            "types={'article_id': 'VARCHAR'})").fetchall())
    rows = []
    for (a, s), t in zip(q[["anchor", "target_slot"]].itertuples(index=False), tt):
        lists = [assoc.get((a, s), np.array([], dtype=int)), t]
        fused = M.rrf([[lists[0]], [lists[1]]], [w, 1 - w], 4 * k)[0].tolist()
        for r, b in enumerate(diversify(fused, ptype, style, per, int(cfg.serving.max_per_product_type)), 1):
            l = lift.get((a, b))
            rows.append((a, s, r, b, "co_purchase" if l else "style_match", l))
    return pd.DataFrame(rows, columns=["anchor", "slot", "rank", "article_id", "source", "lift"])



def main() -> None:
    cfg = load_config()
    ctl = complete_the_look(cfg)
    out = cfg.path("processed") / "models" / "complete_the_look.parquet"
    ctl.to_parquet(out)
    print(f"wrote {out} ({len(ctl):,} rows, {ctl.anchor.nunique():,} anchors)")


if __name__ == "__main__":
    main()

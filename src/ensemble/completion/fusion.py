"""Fixed-fusion Track B baselines (the models Round 3 has to beat).

Imports neither LightGBM nor PyTorch, so the ranker process and the tower process
can both use it. Every list here depends only on (anchor, target_slot): none of
these baselines is personalized, which is exactly the gap Phase 3 closes.

``slot_popularity``     per-slot recent popularity (D-004 reference)
``assoc_npmi``          association rules, support >= 3 and lift > 1, NPMI order, popularity backfill
``shipped_rrf_hybrid``  the Round-1 shipped model: RRF(w = 0.5) of association and the two towers
``rrf_all_sources``     equal-weight RRF over every source in the union (fair-retrieval control)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASELINES = ("slot_popularity", "assoc_npmi", "shipped_rrf_hybrid", "rrf_all_sources")
RRF_C = 60.0


def _grouped(rows, k: int) -> dict:
    out: dict[tuple, list[int]] = {}
    for key1, key2, art in rows:
        lst = out.setdefault((key1, key2), [])
        if len(lst) < k:
            lst.append(art)
    return out


def baseline_lists(con, cfg, uni: pd.DataFrame, tt: pd.DataFrame | None, ttc: pd.DataFrame | None) -> dict:
    """Round-1 style recommendations at (anchor, target_slot) level — none is personalized."""
    k = int(cfg.track_b.k)
    # Round-1 fused lists of length 4k and back-filled each of them from the slot's
    # 8k most popular articles, so popularity earned reciprocal-rank credit inside the
    # association list rather than only filling the tail. Reproduced exactly here.
    pop = {s: g.article_id.to_numpy()[: 8 * k] for s, g in uni.groupby("slot", sort=False)}
    assoc = _grouped(con.execute(f"""
        SELECT src, dst_slot, dst FROM _tb_assoc WHERE co >= 3 AND lift > 1
        ORDER BY src, dst_slot, npmi DESC, dst""").fetchall(), 4 * k)
    tt_lists = {} if tt is None else _grouped(
        tt.sort_values(["anchor", "target_slot", "sim"], ascending=[True, True, False])
        [["anchor", "target_slot", "article_id"]].itertuples(index=False, name=None), 4 * k)
    ttc_lists = {} if ttc is None else _grouped(
        ttc.sort_values(["anchor", "target_slot", "sim"], ascending=[True, True, False])
        [["anchor", "target_slot", "article_id"]].itertuples(index=False, name=None), 4 * k)
    style = _grouped(con.execute("""
        SELECT ar.article_id, y.dst_slot, y.dst FROM _tb_keys kk
        JOIN _tb_attrs ar ON ar.article_id = kk.anchor
        JOIN _tb_style y ON y.src_code = ar.product_code AND y.dst_slot = kk.target_slot
        WHERE y.s_co >= 3 AND y.s_lift > 1
        ORDER BY ar.article_id, y.dst_slot, y.s_npmi DESC, y.dst""").fetchall(), 4 * k)
    return {"pop": pop, "assoc": assoc, "tt": tt_lists, "ttc": ttc_lists, "style": style}


def fill(head: list[int], tail, k: int) -> np.ndarray:
    seen = set(head)
    out = list(head[:k])
    for a in tail:
        if len(out) >= k:
            break
        if a not in seen:
            out.append(a)
            seen.add(a)
    return np.asarray(out[:k], dtype=np.int64)


def rrf(lists: list[list[int]], weights: list[float], k: int) -> list[int]:
    score: dict[int, float] = {}
    for w, lst in zip(weights, lists):
        for r, a in enumerate(lst):
            score[a] = score.get(a, 0.0) + w / (RRF_C + r + 1)
    return sorted(score, key=lambda a: (-score[a], a))[:k]


def baseline_recs(q: pd.DataFrame, base: dict, cfg) -> dict[str, list[np.ndarray]]:
    """Materialise each baseline's top-k for every query row of ``q``."""
    k = int(cfg.track_b.k)
    pop, assoc, tt, ttc, style = base["pop"], base["assoc"], base["tt"], base["ttc"], base["style"]
    cache: dict[tuple, dict[str, np.ndarray]] = {}
    out = {name: [] for name in BASELINES}
    for a, s in zip(q.anchor.to_numpy(), q.target_slot.to_numpy()):
        key = (a, s)
        got = cache.get(key)
        if got is None:
            pl = pop.get(s, np.empty(0, dtype=np.int64))
            al, tl, cl, sl = assoc.get(key, []), tt.get(key, []), ttc.get(key, []), style.get(key, [])
            al4 = fill(al, pl, 4 * k).tolist()     # the Round-1 association channel
            got = {
                "slot_popularity": np.asarray(pl[:k], dtype=np.int64),
                "assoc_npmi": fill(al, pl, k),
                "shipped_rrf_hybrid": fill(rrf([al4, tl], [0.5, 0.5], k), pl, k),
                # Fair-retrieval control: every source contributes its own ranking exactly
                # once (popularity included), with no back-filled duplicates inflating it.
                "rrf_all_sources": fill(rrf([al, sl, tl, cl, list(pl[: 4 * k])], [1.0] * 5, k), pl, k),
            }
            cache[key] = got
        for name in BASELINES:
            out[name].append(got[name])
    return out

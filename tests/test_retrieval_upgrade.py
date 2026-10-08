"""Retrieval-ceiling upgrade: point-in-time pools, merge-identical ranks, superset frontier,
append-only candidate construction."""
import duckdb
import numpy as np
import pandas as pd

from test_retrieval import SQL_CHANNELS, WEEK, cfg_r, make_db

from ensemble.research import retrieval_upgrade as RU

P2 = ["repeat", "pop", "pop_age", "cf", "variant", "dept_pop", "section_pop", "covis", "new_arrival_pers"]


def _pool(future_article: int, extra: bool):
    con = make_db(future_article, extra)
    r = cfg_r(channels=P2)
    return RU.build_pool(con, WEEK, r, {c: 5 for c in P2}, "_users")


def test_pool_ignores_label_week():
    a, b = _pool(99, False), _pool(77, True)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) and {77, 99}.isdisjoint(set(a.article_id))


def test_pool_ranks_equal_the_production_merge():
    from ensemble.candidates.retrieval import build_candidates
    caps = {c: 3 for c in P2}
    con = make_db(99)
    r = cfg_r(channels=P2, **{f"{c}_k": 3 for c in P2})
    build_candidates(con, WEEK, r)
    cand = con.execute("SELECT * FROM cand").df()
    pool = RU.build_pool(make_db(99), WEEK, r, caps, "_users")
    for c in P2:
        merged = cand[cand[f"{c}_rank"].notna()][["customer_idx", "article_id", f"{c}_rank"]]
        merged = merged.astype({f"{c}_rank": int}).sort_values(["customer_idx", "article_id"]).to_numpy().tolist()
        mine = pool[pool.ch == RU.CH_ID[c]][["customer_idx", "article_id", "rnk"]]
        mine = mine.astype(int).sort_values(["customer_idx", "article_id"]).to_numpy().tolist()
        assert merged == mine, c


def test_seeded_frontier_only_grows_and_starts_at_the_seed():
    rng = np.random.default_rng(0)
    pools = [{}]
    uid = 0
    for c in ("a", "b"):
        n = 400
        ranks = np.tile(np.arange(1, 21), n // 20).astype(np.int16)
        ids = np.arange(uid, uid + n)
        uid += n
        pools[0][c] = (ranks, ids, rng.random(n) < (0.05 if c == "a" else 0.02))
    fr = RU.greedy_seeded(pools, 20, int(sum(p[2].sum() for p in pools[0].values())), {"a": 3, "b": 2}, (1, 5))
    assert fr[0]["caps"] == {"a": 3, "b": 2}
    for prev, nxt in zip(fr, fr[1:]):
        assert all(nxt["caps"][c] >= prev["caps"][c] for c in ("a", "b"))
        assert nxt["cand"] >= prev["cand"] and nxt["recall"] >= prev["recall"]


def test_append_only_never_evicts_and_respects_the_budget(tmp_path):
    d = tmp_path
    (d / "pool").mkdir()
    pool = pd.DataFrame({"customer_idx": [1, 1, 1, 2, 2], "article_id": [10, 11, 12, 20, 21],
                         "ch": [0, 0, 7, 1, 1], "rnk": [1, 2, 1, 1, 2], "score": [1.0, 0.5, 0.9, 3.0, 2.0]})
    pool = pool.astype({"customer_idx": "int32", "article_id": "int32", "ch": "int8", "rnk": "int16"})
    pool.to_parquet(d / "pool" / "part-000.parquet", index=False)
    deep = pd.DataFrame({"customer_idx": [1, 1, 1, 1, 2, 2], "rnk": [1, 2, 3, 4, 1, 2],
                         "article_id": [11, 30, 31, 32, 21, 40], "score": [9, 8, 7, 6, 5, 4.0]})
    deep.astype({"customer_idx": "int32", "rnk": "int16", "article_id": "int32"}).to_parquet(
        d / "deep_sasrec.parquet", index=False)
    con = duckdb.connect()
    con.execute("CREATE TABLE _u AS SELECT * FROM (VALUES (1), (2)) t(customer_idx)")
    caps = {"repeat": 2, "pop": 1, "covis": 1}
    info = RU.build_cand(con, d, {"caps": caps, "append": ["sasrec"], "budget": 5}, "_u")
    c = con.execute("SELECT * FROM cand ORDER BY customer_idx, article_id").df()
    got = {k: sorted(g.article_id) for k, g in c.groupby("customer_idx")}
    # customer 1: union {10, 11, 12} kept, 11 not duplicated, then 30, 31 up to 5; customer 2: pop cap 1 -> {20},
    # then 21 (deep rank 1; beyond the pop cap so not in the union) and 40.
    assert got == {1: [10, 11, 12, 30, 31], 2: [20, 21, 40]}
    assert info["n_union"] == 4 and info["n_appended"] == 4
    app = c[c.article_id.isin([30, 31, 21, 40])]
    assert app[[f"{x}_rank" for x in RU.PHASE2_CHANNELS]].isna().all().all()
    assert not c.duplicated(["customer_idx", "article_id"]).any()

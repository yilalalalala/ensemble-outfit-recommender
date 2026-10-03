"""Track B Round 3: protocol, leakage, determinism, metric and fold-boundary tests.

The synthetic database mirrors ``tests/test_leakage.py``: everything inside the
label week is swapped for different articles and the outputs must not move.
"""
from datetime import date

import duckdb
import numpy as np
import pandas as pd
import pytest

from ensemble.completion import association as A
from ensemble.completion import candidates as C
from ensemble.completion import features as FE
from ensemble.completion import metrics as MET
from ensemble.completion import protocol as P
from ensemble.completion import ranker as R
from ensemble.config import Config, load_config
from ensemble.data.splits import Week, make_splits

WEEK = Week(date(2020, 9, 16), date(2020, 9, 22))


def tb_cfg(**over) -> Config:
    base = load_config()
    return Config({**base, "track_b": {**dict(base.track_b), **over}})


def make_db(future_article: int, future_customer: int = 9):
    """Baskets before the week, plus baskets inside it built from ``future_article``."""
    con = duckdb.connect()
    con.execute("""CREATE TABLE articles AS SELECT * FROM (VALUES
        (1, '010', 'upper', 100, 11, 1, '09', 1, 2, 300, 'A', 1, 11, 1001, FALSE),
        (2, '020', 'lower', 200, 12, 2, '10', 2, 3, 301, 'A', 1, 11, 1002, FALSE),
        (3, '030', 'shoes', 300, 13, 3, '11', 3, 4, 302, 'B', 2, 12, 1003, FALSE),
        (4, '040', 'accessories', 400, 14, 4, '12', 1, 2, 303, 'B', 2, 12, 1004, TRUE),
        (77, '770', 'lower', 200, 12, 2, '10', 2, 3, 301, 'A', 1, 11, 1002, FALSE),
        (99, '990', 'lower', 200, 12, 2, '10', 2, 3, 301, 'A', 1, 11, 1002, FALSE))
        t(article_id, product_code, slot, product_type_no, graphical_appearance_no, colour_group_code_i,
          colour_group_code, perceived_colour_value_id, perceived_colour_master_id, department_no,
          index_code, index_group_no, section_no, garment_group_no, is_jewellery)""")
    con.execute("""CREATE TABLE transactions (t_dat DATE, customer_idx INT, article_id INT,
                   price FLOAT, sales_channel_id TINYINT)""")
    rows = []
    for c in range(5):                       # pre-week cross-slot baskets: 1 with 2, 1 with 3
        rows += [("2020-09-01", c, 1), ("2020-09-01", c, 2), ("2020-09-08", c, 1), ("2020-09-08", c, 3)]
    for c in range(5, 8):                    # label-week baskets (the queries)
        rows += [("2020-09-17", c, 1), ("2020-09-17", c, 2)]
    for _ in range(4):                       # label-week-only sales of the swapped article
        rows.append(("2020-09-18", future_customer, future_article))
    for d, c, a in rows:
        con.execute("INSERT INTO transactions VALUES (?, ?, ?, 0.1, 2)", [d, c, a])
    return con


# ---------------------------------------------------------------- leakage


def test_mining_ignores_label_week():
    cfg = tb_cfg(min_support=1)

    def pairs(fa):
        con = make_db(fa)
        A.mine(con, WEEK, cfg)
        return con.execute("SELECT src, dst, co, lift, npmi FROM tb_pairs ORDER BY ALL").fetchall()

    assert pairs(99) == pairs(77)
    # article 1 is in all 10 mined baskets, article 2 in 5 of them, and they co-occur 5 times:
    # lift = 5 * 10 / (10 * 5) = 1, so PMI = 0 and NPMI = 0.
    got = {(src, dst): (co, lift, npmi) for src, dst, co, lift, npmi in pairs(99)}
    assert got[(1, 2)] == (5, pytest.approx(1.0), pytest.approx(0.0))
    assert (1, 99) not in got and (1, 77) not in got


def test_eligible_universe_ignores_label_week():
    cfg = tb_cfg()

    def uni(fa):
        con = make_db(fa)
        u = P.eligible_universe(con, WEEK, cfg)
        return sorted(u.article_id.tolist())

    assert uni(99) == uni(77)
    assert 99 not in uni(99) and 77 not in uni(77)


def test_oracle_universe_does_see_label_week():
    """The Round-1 definition is retained only as an explicit oracle; it must differ."""
    cfg = tb_cfg()
    con = make_db(99)
    honest = set(P.eligible_universe(con, WEEK, cfg).article_id)
    oracle = set(P.eligible_universe(con, WEEK, cfg, oracle=True).article_id)
    assert 99 in oracle and 99 not in honest


def test_features_ignore_label_week():
    """Every feature column is identical when label-week rows are swapped."""
    cfg = tb_cfg(min_support=1, candidates={"assoc_co": 10, "assoc_npmi": 10, "style_co": 10,
                                            "style_npmi": 10, "two_tower": 0, "two_tower_content": 0,
                                            "slot_pop": 10})

    def feats(fa):
        con = make_db(fa)
        A.mine(con, WEEK, cfg)
        uni = P.eligible_universe(con, WEEK, cfg)
        q = P.restrict_truth(P.queries(con, WEEK, cfg), uni)
        C.register(con, q, uni)
        C.build_sources(con, cfg)
        FE.build_context(con, WEEK, cfg, uni, q, None)
        df = FE.chunk_features(con, cfg, 0, 1000, with_labels=True, personalize=True)
        return df.drop(columns=["label"]).sort_values(["qid", "article_id"]).reset_index(drop=True)

    a, b = feats(99), feats(77)
    assert list(a.columns) == list(b.columns)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) > 0


def test_customer_history_excludes_label_week():
    """A purchase inside the label week must not appear in the customer's affinities."""
    cfg = tb_cfg(min_support=1, candidates={"assoc_co": 10, "assoc_npmi": 10, "style_co": 10,
                                            "style_npmi": 10, "two_tower": 0, "two_tower_content": 0,
                                            "slot_pop": 10})
    con = make_db(99)
    A.mine(con, WEEK, cfg)
    uni = P.eligible_universe(con, WEEK, cfg)
    q = P.restrict_truth(P.queries(con, WEEK, cfg), uni)
    C.register(con, q, uni)
    C.build_sources(con, cfg)
    FE.build_context(con, WEEK, cfg, uni, q, None)
    # Customers 5..7 only ever bought inside the label week, so they have no history.
    hist = con.execute("SELECT count(*) FROM _tb_hist WHERE customer_idx >= 5").fetchone()[0]
    assert hist == 0
    df = FE.chunk_features(con, cfg, 0, 1000, with_labels=True, personalize=True)
    assert (df.c_has_history == 0).all()


# ---------------------------------------------------------------- folds and boundaries


def test_folds_end_at_validation_and_never_touch_test():
    splits = make_splits(date(2020, 9, 22), window_weeks=16)
    weeks = P.folds(splits, 6, train_weeks=2)
    assert len(weeks) == 8
    assert weeks[-1] == splits.val
    assert all(w.end < splits.test.start for w in weeks)
    assert weeks == sorted(weeks, key=lambda w: w.start)
    assert all((weeks[i + 1].start - weeks[i].start).days == 7 for i in range(len(weeks) - 1))


def test_training_weeks_precede_each_fold():
    splits = make_splits(date(2020, 9, 22), window_weeks=16)
    train_weeks, n_folds = 2, 6
    weeks = P.folds(splits, n_folds, train_weeks)
    for i in range(train_weeks, len(weeks)):
        used = weeks[i - train_weeks:i]
        assert all(w.end < weeks[i].start for w in used)


def test_query_truth_stays_inside_the_basket():
    cfg = tb_cfg()
    con = make_db(99)
    q = P.queries(con, WEEK, cfg)
    assert len(q) > 0
    for r in q.itertuples():
        same_day = con.execute("SELECT article_id FROM transactions WHERE customer_idx = ? AND t_dat = ?",
                               [r.customer_idx, r.t_dat]).fetchall()
        assert set(np.asarray(r.truth).tolist()) <= {a for (a,) in same_day}


def test_restrict_truth_drops_ineligible_articles():
    cfg = tb_cfg()
    con = make_db(99)
    uni = P.eligible_universe(con, WEEK, cfg)
    q = P.queries(con, WEEK, cfg)
    r = P.restrict_truth(q, uni.iloc[:0])
    assert len(r) == 0 and r.attrs["n_truth_dropped"] == sum(len(t) for t in q.truth.values)


def test_sample_queries_is_deterministic_and_customer_level():
    q = pd.DataFrame({"qid": range(200), "customer_idx": np.repeat(np.arange(50), 4)})
    a = P.sample_queries(q, 80)
    b = P.sample_queries(q, 80)
    pd.testing.assert_frame_equal(a, b)
    assert set(a.customer_idx) <= set(q.customer_idx)
    for c in a.customer_idx.unique():        # a sampled customer keeps all of their queries
        assert (a.customer_idx == c).sum() == (q.customer_idx == c).sum()


# ---------------------------------------------------------------- metrics


def _uni(n=20):
    return pd.DataFrame({"article_id": np.arange(1, n + 1), "slot": "upper", "is_jewellery": False,
                         "pop_recent": np.arange(n, 0, -1), "pop_window": np.arange(n, 0, -1),
                         "days_sold_window": 5, "days_since_last_sale": 1,
                         "age_days": np.r_[np.full(n - 2, 400), [10, 10]], "mean_price": 0.1})


def test_recall_and_ndcg_match_hand_computation():
    cfg = tb_cfg(k=4, head_share=0.5, new_item_days=28)
    uni = _uni()
    q = pd.DataFrame({"qid": [0], "customer_idx": [1], "target_slot": ["upper"], "truth": [np.array([3, 7])]})
    m = MET.evaluate([np.array([5, 3, 9, 7])], q, uni, cfg)
    assert m["recall@4"] == pytest.approx(1.0)
    assert m["recall@5"] == pytest.approx(1.0)       # both hits are inside the top 4
    idcg = 1 / np.log2(2) + 1 / np.log2(3)
    dcg = 1 / np.log2(3) + 1 / np.log2(5)            # hits at ranks 2 and 4 (0-indexed 1 and 3)
    assert m["ndcg@4"] == pytest.approx(dcg / idcg)


def test_recall_at_5_is_a_prefix_of_recall_at_12():
    cfg = tb_cfg(k=12, head_share=0.5)
    uni = _uni()
    q = pd.DataFrame({"qid": [0], "customer_idx": [1], "target_slot": ["upper"], "truth": [np.array([1, 20])]})
    m = MET.evaluate([np.arange(1, 13)], q, uni, cfg)
    assert m["recall@5"] == pytest.approx(0.5) and m["recall@12"] == pytest.approx(0.5)


def test_new_item_segment_counts_recently_launched_articles():
    cfg = tb_cfg(k=4, head_share=0.5, new_item_days=28)
    uni = _uni()
    seg = MET.segments(uni, cfg)
    assert seg["new_item"] == {19, 20}
    q = pd.DataFrame({"qid": [0], "customer_idx": [1], "target_slot": ["upper"], "truth": [np.array([19, 20])]})
    m = MET.evaluate([np.array([19, 1, 2, 3])], q, uni, cfg)
    assert m["recall@4_new_item"] == pytest.approx(0.5)
    assert m["n_truth_new_item"] == 2


def test_diversity_and_coverage():
    cfg = tb_cfg(k=4, head_share=0.5)
    uni = _uni()
    attrs = pd.DataFrame({"article_id": np.arange(1, 21), "product_type_no": [1] * 10 + [2] * 10,
                          "product_code": [f"c{i // 2}" for i in range(20)]})
    q = pd.DataFrame({"qid": [0], "customer_idx": [1], "target_slot": ["upper"], "truth": [np.array([1])]})
    m = MET.evaluate([np.array([1, 2, 3, 4])], q, uni, cfg, attrs=attrs)
    assert m["diversity_product_type@4"] == pytest.approx(0.25)   # all four share product type 1
    assert m["diversity_product_code@4"] == pytest.approx(0.5)    # two styles among four items
    assert m["catalog_coverage@4"] == pytest.approx(4 / 20)


def test_cluster_bootstrap_point_estimate_matches_the_metric():
    rng = np.random.default_rng(0)
    cust = rng.integers(0, 40, 400)
    a, b = rng.random(400), rng.random(400)
    out = MET.cluster_bootstrap(cust, a, b, n_boot=200)
    assert out["diff"] == pytest.approx(b.mean() - a.mean())
    assert out["ci95"][0] < out["diff"] < out["ci95"][1]
    assert out["n_customers"] == len(np.unique(cust))


def test_cluster_bootstrap_is_wider_than_ignoring_clusters():
    """Correlated queries inside a customer must widen the interval."""
    rng = np.random.default_rng(1)
    per_customer = rng.normal(0.05, 0.1, 50)
    cust = np.repeat(np.arange(50), 20)
    a = np.zeros(1000)
    b = np.repeat(per_customer, 20)
    clustered = MET.cluster_bootstrap(cust, a, b, n_boot=500)
    naive = MET.cluster_bootstrap(np.arange(1000), a, b, n_boot=500)
    width = lambda d: d["ci95"][1] - d["ci95"][0]  # noqa: E731
    assert width(clustered) > 3 * width(naive)


# ---------------------------------------------------------------- ranker plumbing


def test_groups_and_top_k_are_deterministic():
    df = pd.DataFrame({"qid": [0, 0, 0, 1, 1], "article_id": [5, 2, 9, 4, 1]})
    assert R.groups_of(df.qid.to_numpy()).tolist() == [3, 2]
    top = R.top_k(df, np.array([1.0, 1.0, 0.5, 0.1, 0.2]), k=2)
    assert top[0].tolist() == [2, 5]          # tie on score breaks on the smaller article_id
    assert top[1].tolist() == [1, 4]


def test_drop_personalization_removes_only_customer_features():
    feats = ["a_npmi", "c_n_tx", "cu_type_share", "x_npmi_type", "cand_pop_recent"]
    assert FE.drop_personalization(feats) == ["a_npmi", "cand_pop_recent"]


def test_cache_key_changes_with_candidate_budget():
    from ensemble.completion.pipeline import cache_key
    a = cache_key(tb_cfg())
    b = cache_key(tb_cfg(candidates={**dict(load_config().track_b.candidates), "slot_pop": 7}))
    assert a != b
    assert cache_key(tb_cfg()) == a


# ---------------------------------------------------------------- stronger leakage probes


def make_db_volume(n_label_baskets: int):
    """Same pre-week history, a varying number of label-week baskets of articles 1 and 2.

    The swap test above proves no *new* article leaks in. This proves that the
    *volume* of label-week activity for articles that already exist cannot move a
    single feature: article 2's popularity, recency, price and the 1->2 co-count
    all have to come from before the cutoff only.
    """
    con = duckdb.connect()
    con.execute("""CREATE TABLE articles AS SELECT * FROM (VALUES
        (1, '010', 'upper', 100, 11, 1, '09', 1, 2, 300, 'A', 1, 11, 1001, FALSE),
        (2, '020', 'lower', 200, 12, 2, '10', 2, 3, 301, 'A', 1, 11, 1002, FALSE),
        (3, '030', 'shoes', 300, 13, 3, '11', 3, 4, 302, 'B', 2, 12, 1003, FALSE),
        (4, '040', 'accessories', 400, 14, 4, '12', 1, 2, 303, 'B', 2, 12, 1004, TRUE))
        t(article_id, product_code, slot, product_type_no, graphical_appearance_no, colour_group_code_i,
          colour_group_code, perceived_colour_value_id, perceived_colour_master_id, department_no,
          index_code, index_group_no, section_no, garment_group_no, is_jewellery)""")
    con.execute("""CREATE TABLE transactions (t_dat DATE, customer_idx INT, article_id INT,
                   price FLOAT, sales_channel_id TINYINT)""")
    rows = []
    for c in range(5):
        rows += [("2020-09-01", c, 1), ("2020-09-01", c, 2), ("2020-09-08", c, 1), ("2020-09-08", c, 3)]
    # The two queried label-week baskets are always present; the rest only add volume.
    for c in range(5, 5 + n_label_baskets):
        rows += [("2020-09-17", c, 1), ("2020-09-17", c, 2)]
    for d, c, a in rows:
        con.execute("INSERT INTO transactions VALUES (?, ?, ?, 0.1, 2)", [d, c, a])
    return con


def _feat_cfg():
    return tb_cfg(min_support=1, candidates={"assoc_co": 10, "assoc_npmi": 10, "style_co": 10,
                                             "style_npmi": 10, "two_tower": 0, "two_tower_content": 0,
                                             "slot_pop": 10})


def test_label_week_volume_does_not_move_any_feature():
    cfg = _feat_cfg()

    def feats(n):
        con = make_db_volume(n)
        A.mine(con, WEEK, cfg)
        uni = P.eligible_universe(con, WEEK, cfg)
        q = P.restrict_truth(P.queries(con, WEEK, cfg), uni)
        C.register(con, q, uni)
        C.build_sources(con, cfg)
        FE.build_context(con, WEEK, cfg, uni, q, None)
        df = FE.chunk_features(con, cfg, 0, 1000, with_labels=True, personalize=True)
        # Customers 5 and 6 exist in both databases; compare only their rows.
        keep = q.qid[q.customer_idx.isin([5, 6])]
        return (df[df.qid.isin(keep)].drop(columns=["qid", "customer_idx"])
                .sort_values(["article_id"]).reset_index(drop=True))

    pd.testing.assert_frame_equal(feats(2), feats(20))
    assert len(feats(2)) > 0


def test_label_week_volume_does_not_move_the_eligible_catalogue_or_mining():
    cfg = tb_cfg(min_support=1)

    def mined(n):
        con = make_db_volume(n)
        A.mine(con, WEEK, cfg)
        pairs = con.execute("SELECT src, dst, co, lift, npmi FROM tb_pairs ORDER BY ALL").fetchall()
        uni = P.eligible_universe(con, WEEK, cfg)
        return pairs, uni.sort_values("article_id").reset_index(drop=True)

    (pa, ua), (pb, ub) = mined(2), mined(20)
    assert pa == pb
    pd.testing.assert_frame_equal(ua, ub)


def test_test_mode_training_weeks_precede_the_test_week():
    """``backtest.run(mode="test")`` trains on the two weeks before test, never on test."""
    splits = make_splits(date(2020, 9, 22), window_weeks=16)
    train_weeks = 2
    weeks = [splits.test.shift(-k) for k in range(train_weeks, -1, -1)]
    assert weeks[-1] == splits.test
    assert [w for w in weeks[:-1]] == [splits.val.shift(-1), splits.val]
    assert all(w.end < splits.test.start for w in weeks[:-1])


# ---------------------------------------------------------------- feature-group ablations


def test_every_ablation_group_removes_the_features_it_names():
    feats = ["a_co", "a_co_d14", "a_co_d14_ratio", "a_lift", "a_npmi", "a_pmi", "a_n_src", "a_n_dst",
             "a_rank_co", "a_rank_npmi", "a_last_co_days", "a_co_share", "a_co_d56", "a_co_d56_ratio",
             "has_article_evidence", "has_style_evidence", "backoff_level",
             "s_co", "s_co_d14", "s_co_d56", "s_co_share", "s_last_co_days", "s_lift", "s_n_dst",
             "s_n_src", "s_npmi", "s_pmi", "s_rank_co", "s_rank_npmi",
             "src_two_tower", "src_two_tower_content", "tt_rank", "ttc_rank", "tt_pair_sim",
             "ttc_pair_sim", "clip_sim", "src_slot_pop",
             "cand_pop_recent", "cand_pop_window", "cand_days_sold", "cand_days_since_last_sale",
             "cand_age_days", "cand_mean_price", "cand_is_jewellery", "cand_slot_pop_rank",
             "cand_slot_pop_pct", "cand_log_pop_recent", "cand_log_pop_window",
             "c_has_history", "c_n_tx", "c_n_days", "c_days_since_last", "c_mean_price",
             "cu_article_n", "cu_article_days_ago", "cu_style_n", "cu_type_share", "cu_dept_share",
             "cu_section_share", "cu_ggroup_share", "cu_colour_share", "cu_cmaster_share",
             "cu_slot_share", "cu_type_n", "cu_dept_n", "cu_price_dist",
             "x_npmi_type", "x_tt_type", "x_co_dept",
             "same_colour_group", "same_colour_master", "same_colour_value", "same_index_group",
             "same_section", "same_department", "same_garment_group", "same_graphical",
             "same_product_type", "price_log_ratio", "price_log_ratio_abs", "price_diff",
             "cand_price_tier", "anchor_price_tier", "price_tier_diff",
             "cand_product_type_no", "target_slot_id", "anchor_slot_id"]
    # check_groups is the contract the backtest enforces on the real schema.
    counts = FE.check_groups(feats)
    assert counts == FE.GROUP_MIN_FEATURES
    for group in FE.GROUPS:
        kept = FE.subset(feats, (group,))
        assert len(kept) < len(feats), group
    # "minus towers" must leave no tower or CLIP-pair signal behind, interactions included.
    no_towers = FE.subset(feats, ("towers",))
    assert not [f for f in no_towers if "tt" in f]
    # "minus association" must drop the article-level evidence and its interactions, and keep style.
    no_assoc = FE.subset(feats, ("association",))
    assert not [f for f in no_assoc if f.startswith(("a_", "x_npmi", "x_co"))]
    assert "s_npmi" in no_assoc and "has_style_evidence" in no_assoc
    # "minus popularity" must leave no candidate popularity or age feature.
    no_pop = FE.subset(feats, ("popularity",))
    assert not [f for f in no_pop if "pop" in f or f == "cand_age_days"]


def test_subset_rejects_a_group_that_removes_nothing():
    """A renamed feature must fail loudly instead of producing a no-op ablation."""
    with pytest.raises(ValueError, match="removed nothing"):
        FE.subset(["a_npmi", "cand_pop_recent"], ("towers",))


def test_check_groups_rejects_a_schema_that_lost_a_group():
    with pytest.raises(ValueError, match="no longer match the schema"):
        FE.check_groups(["a_npmi", "cand_pop_recent"])


def test_ablation_subsets_names_and_rejects_unknown_groups():
    feats = ["a_npmi", "c_n_tx", "cu_article_n", "cu_type_share", "x_tt_type", "tt_rank",
             "tt_pair_sim", "ttc_rank", "ttc_pair_sim", "src_two_tower", "src_two_tower_content",
             "cand_pop_recent"]
    cfg = tb_cfg(feature_ablations=["repeat", "towers"])
    out = FE.ablation_subsets(cfg, feats)
    assert set(out) == {"lgbm_personalized", "lgbm_compatibility",
                        "lgbm_minus_repeat", "lgbm_minus_towers"}
    assert out["lgbm_personalized"] == feats
    assert "cu_article_n" not in out["lgbm_minus_repeat"]
    assert "cu_type_share" in out["lgbm_minus_repeat"]
    with pytest.raises(KeyError, match="unknown feature ablation group"):
        FE.ablation_subsets(tb_cfg(feature_ablations=["not_a_group"]), feats)


def test_combined_ablation_group_removes_both_groups():
    feats = ["price_log_ratio", "cand_price_tier", "anchor_price_tier", "price_diff",
             "price_log_ratio_abs", "price_tier_diff", "cu_price_dist",
             "same_colour_group", "same_colour_master", "same_colour_value", "a_npmi"]
    cfg = tb_cfg(feature_ablations=["price+colour"])
    out = FE.ablation_subsets(cfg, feats)
    assert out["lgbm_minus_price_colour"] == ["a_npmi"]


def test_group_gain_attributes_split_gain_to_groups():
    imp = [("cu_article_n", 0.4), ("a_npmi", 0.3), ("tt_pair_sim", 0.2), ("target_slot_id", 0.1)]
    g = R.group_gain(imp)
    assert g["repeat"] == pytest.approx(0.4)
    assert g["personalization"] == pytest.approx(0.4)
    assert g["association"] == pytest.approx(0.3)
    assert g["towers"] == pytest.approx(0.2)
    assert g["other"] == pytest.approx(0.1)        # the slot id belongs to no group


# ---------------------------------------------------------------- fixed-fusion baselines


def test_rrf_scores_and_fill_are_deterministic_and_rank_correctly():
    from ensemble.completion import fusion as FU
    a, b = [10, 20, 30], [30, 40]
    fused = FU.rrf([a, b], [1.0, 1.0], 4)
    assert fused[0] == 30                    # the only article in both lists wins
    assert set(fused) == {10, 20, 30, 40}
    assert FU.rrf([a, b], [1.0, 1.0], 4) == fused
    assert FU.fill([1, 2], [2, 3, 4], 4).tolist() == [1, 2, 3, 4]      # no duplicate 2
    assert FU.fill([1, 2, 3, 4, 5], [9], 3).tolist() == [1, 2, 3]      # head already full


def test_rrf_from_ranks_matches_the_hand_computation():
    from ensemble.completion.backtest import RRF_SOURCES, rrf_from_ranks
    from ensemble.completion.fusion import RRF_C
    cols = [c for c, _ in RRF_SOURCES]
    df = pd.DataFrame({c: [np.nan] * 2 for c in cols})
    df.loc[0, cols[0]] = 1.0
    df.loc[1, cols[0]] = 1.0
    df.loc[1, cols[1]] = 3.0
    s = rrf_from_ranks(df)
    assert s[0] == pytest.approx(1 / (RRF_C + 1))
    assert s[1] == pytest.approx(1 / (RRF_C + 1) + 1 / (RRF_C + 3))
    assert s[1] > s[0]


def test_baseline_recs_are_capped_at_k_and_unique():
    from ensemble.completion import fusion as FU
    cfg = tb_cfg(k=4)
    base = {"pop": {"lower": np.array([7, 8, 9, 10, 11, 12])},
            "assoc": {(1, "lower"): [9, 7]}, "tt": {(1, "lower"): [8]},
            "ttc": {}, "style": {(1, "lower"): [12]}}
    q = pd.DataFrame({"anchor": [1, 1], "target_slot": ["lower", "lower"]})
    out = FU.baseline_recs(q, base, cfg)
    for name in FU.BASELINES:
        assert len(out[name]) == 2
        for r in out[name]:
            assert len(r) == len(set(r.tolist())) <= 4
        assert np.array_equal(out[name][0], out[name][1])     # same key -> same list
    assert out["slot_popularity"][0].tolist() == [7, 8, 9, 10]

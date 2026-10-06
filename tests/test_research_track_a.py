"""Track A research benchmark: protocol invariants, point-in-time model inputs, faithful
baseline behaviour on toy data, CPU reproducibility, tie-breaking and the cluster bootstrap."""
from datetime import date

import duckdb
import numpy as np
import pandas as pd
import pytest

from ensemble.config import load_config
from ensemble.data.splits import Week
from ensemble.research import protocol as P

WEEK = Week(date(2020, 9, 16), date(2020, 9, 22))


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

def research_cfg():
    return load_config("track_a_research")


def test_protocol_is_frozen_and_leak_free():
    cfg = research_cfg()
    P.check_protocol(cfg)
    folds, tuning = P.reporting_folds(cfg), P.tuning_folds(cfg)
    conf = P.week_of(cfg.research.protocol.confirmation_week)
    assert len(folds) == 6 and folds[-1].start == date(2020, 9, 9)
    assert all(f.end < conf.start for f in folds)
    assert all(t.end < folds[0].start for t in tuning)
    # Every per-week model the ensemble needs exists before its fold and never reaches the confirmation week.
    assert all(w.end < conf.start for w in P.all_model_weeks(cfg))
    for f in folds:
        tw = P.ranker_train_weeks(cfg, f)
        assert len(tw) == int(cfg.ranker.train_weeks) and all(w.end < f.start for w in tw)


def test_restrict_and_fill_respects_eligibility_order_and_k():
    out = P.restrict_and_fill({1: [5, 9, 5, 7], 2: []}, [1, 2, 3], {5, 7, 8, 1, 2}, lambda u: [8, 5, 1, 2], k=3)
    assert out[1] == [5, 7, 8]          # 9 ineligible, duplicate 5 dropped, back-filled
    assert out[2] == [8, 5, 1] and out[3] == [8, 5, 1]


def test_top_k_breaks_ties_on_smaller_article_id():
    ids = np.array([10, 20, 30, 40, 50])
    S = np.array([[1.0, 3.0, 3.0, 3.0, 0.0],      # three-way tie straddling the k=2 boundary
                  [5.0, 4.0, 3.0, 2.0, 1.0]])
    items, scores = P.top_k_from_scores(np.arange(2), ids, S, 2)
    assert items.tolist() == [[20, 30], [10, 20]]
    assert scores.tolist() == [[3.0, 3.0], [5.0, 4.0]]


# ---------------------------------------------------------------------------
# Point-in-time inputs (label-week corruption)
# ---------------------------------------------------------------------------

def make_db(future_article: int, extra_future_rows: int = 5):
    con = duckdb.connect()
    con.execute("""CREATE TABLE transactions (t_dat DATE, customer_idx INT, article_id INT,
                   price FLOAT, sales_channel_id TINYINT)""")
    rows = [("2020-08-01", 0, 3), ("2020-09-01", 0, 1), ("2020-09-10", 0, 2), ("2020-09-12", 0, 4),
            ("2020-09-14", 1, 1), ("2020-09-15", 1, 3), ("2020-09-13", 1, 2), ("2020-09-15", 2, 5),
            ("2020-09-14", 2, 1), ("2020-09-15", 2, 4), ("2020-09-11", 1, 4), ("2020-09-12", 2, 2)]
    con.executemany("INSERT INTO transactions VALUES (?, ?, ?, 0.1, 2)", rows)
    for c in (0, 1, 2, 3):
        for _ in range(extra_future_rows):
            con.execute("INSERT INTO transactions VALUES ('2020-09-17', ?, ?, 0.1, 2)", [c, future_article])
    return con


def test_interactions_and_sequences_ignore_label_week():
    from ensemble.research import data as D

    def inputs(future_article, n):
        con = make_db(future_article, n)
        x = D.interactions(con, WEEK, window_weeks=26, min_item_count=1)
        v = D.vocabulary(con, WEEK, vocab_weeks=26, min_item_count=1)
        s = D.sequences(con, WEEK, v, history_weeks=104, max_len=4)
        return x, v, s, P.eligible(con, WEEK, 28)

    a, b = inputs(99, 5), inputs(77, 50)
    for k in ("users", "items", "u", "i"):
        assert np.array_equal(a[0][k], b[0][k])
    assert np.array_equal(a[1], b[1])
    for k in ("users", "seq", "last_day"):
        assert np.array_equal(a[2][k], b[2][k])
    assert np.array_equal(a[3], b[3])
    assert 99 not in a[1] and 77 not in b[1] and 3 not in a[0]["users"]   # customer 3 only buys in the label week


def test_sequences_are_chronological_and_right_aligned():
    from ensemble.research import data as D
    con = make_db(99)
    v = D.vocabulary(con, WEEK, 26, 1)
    s = D.sequences(con, WEEK, v, 104, max_len=5)
    tok = {int(a): i + 1 for i, a in enumerate(v)}
    row = s["seq"][list(s["users"]).index(0)]
    # customer 0: 3 (08-01), 1 (09-01), 2 (09-10), 4 (09-12) -> oldest first, left-padded
    assert row.tolist() == [0, tok[3], tok[1], tok[2], tok[4]]
    assert s["last_day"][list(s["users"]).index(0)] == 4   # 2020-09-12 is 4 days before 09-16


# ---------------------------------------------------------------------------
# Faithful behaviour on toy data, and CPU reproducibility
# ---------------------------------------------------------------------------

def toy_graph():
    """Two communities: users 0-19 buy items 0-9, users 20-39 buy items 10-19; one item per user held out."""
    rng = np.random.default_rng(0)
    u, i, held = [], [], {}
    for user in range(40):
        pool = np.arange(10) if user < 20 else np.arange(10, 20)
        items = rng.choice(pool, 6, replace=False)
        held[user] = int(items[0])
        u += [user] * 5
        i += items[1:].tolist()
    return np.array(u, np.int32), np.array(i, np.int32), held


@pytest.mark.parametrize("model", ["bpr", "lightgcn"])
def test_pairwise_models_learn_communities_and_are_cpu_reproducible(model):
    import torch

    from ensemble.research import models as M
    u, i, held = toy_graph()
    # Toy-sized data needs real regularisation; at reg 1e-4 MF memorises (train loss ~0.003).
    p = {"lr": 0.05, "reg": 0.05 if model == "bpr" else 0.01, "batch_size": 64, "epochs": 100}

    def train():
        M.seed_everything(0)
        torch.set_num_threads(1)
        if model == "bpr":
            net = M.BPRMF(40, 20, 16)
        else:
            R, Rt = M.normalized_graph(u, i, 40, 20, torch.device("cpu"))
            net = M.LightGCN(40, 20, 16, 2, R, Rt)
        log = M.train_pairwise(net, u, i, 20, p, 0, torch.device("cpu"))
        U, I = (x.numpy() for x in net.final())
        return U, I, log

    U, I, log = train()
    assert log[-1]["loss"] < log[0]["loss"]
    S = U @ I.T
    own = [S[user, (np.arange(10) if user < 20 else np.arange(10, 20))].mean() for user in range(40)]
    other = [S[user, (np.arange(10, 20) if user < 20 else np.arange(10))].mean() for user in range(40)]
    assert np.mean(np.array(own) > np.array(other)) == 1.0
    # The held-out item (never trained on for that user) outranks the other community's items:
    # mean AUC of held-out vs other-community items.
    auc = [np.mean(S[user, held[user]] > S[user, (np.arange(10, 20) if user < 20 else np.arange(10))]) for user in range(40)]
    assert np.mean(auc) >= (0.85 if model == "bpr" else 0.95)   # graph smoothing generalises better than MF
    U2, I2, _ = train()
    assert np.array_equal(U, U2) and np.array_equal(I, I2)      # identical CPU seed/config -> identical output


@pytest.mark.parametrize("loss", ["gbce", "ce", "bce"])
def test_sasrec_learns_a_deterministic_transition_and_is_cpu_reproducible(loss):
    import torch

    from ensemble.research import models as M
    V, L = 12, 6
    rng = np.random.default_rng(0)
    seqs = []
    for _ in range(400):                       # item t is always followed by t + 1 (cyclic)
        start, n = rng.integers(1, V + 1), rng.integers(3, L + 2)
        toks = [(start - 1 + j) % V + 1 for j in range(n)]
        seqs.append([0] * (L + 1 - n) + toks)
    seq = np.array(seqs, np.int32)
    p = {"lr": 0.01, "batch_size": 64, "epochs": 25, "loss": loss, "negatives": 4, "gbce_t": 0.75}

    def train():
        M.seed_everything(0)
        torch.set_num_threads(1)
        net = M.SASRec(V, L, 16, 1, 1, 0.0)
        log = M.train_sasrec(net, seq, V, p, 0, torch.device("cpu"))
        net.eval()
        q = np.array([[0, 0, 0, 3, 4, 5], [0, 0, 0, 0, 10, 11]], np.int64)
        h = net.represent(torch.from_numpy(q)).numpy()
        return h @ net.item.weight[1:].detach().numpy().T, log

    scores, log = train()
    assert log[-1]["loss"] < log[0]["loss"]
    assert scores[0].argmax() + 1 == 6 and scores[1].argmax() + 1 == 12
    scores2, _ = train()
    assert np.allclose(scores, scores2, atol=1e-6)


def test_negative_sampler_never_returns_a_training_positive():
    from ensemble.research.models import NegativeSampler
    u = np.array([0, 0, 0, 1], np.int32)
    i = np.array([0, 1, 2, 3], np.int32)
    s = NegativeSampler(u, i, 5, seed=0)
    j = s.sample(np.zeros(1000, np.int64))
    assert set(j.tolist()) <= {3, 4}


# ---------------------------------------------------------------------------
# Metrics and the bootstrap
# ---------------------------------------------------------------------------

def test_ap_ndcg_match_competition_definitions():
    from ensemble.research.evaluate import ap_at_k, ndcg_at_k
    assert ap_at_k([1, 2, 3], {1, 3}, 12) == pytest.approx((1 / 1 + 2 / 3) / 2)
    assert ap_at_k([9] * 3, {1}, 12) == 0.0
    assert ndcg_at_k([2, 1], {1}, 12) == pytest.approx(1 / np.log2(3))


def test_cluster_bootstrap_clusters_customers_across_folds_and_resamples_the_ratio():
    from ensemble.research.evaluate import cluster_bootstrap
    rng = np.random.default_rng(0)
    rows = [(c, f) for c in range(300) for f in range(3) if rng.random() < 0.7]
    a = pd.DataFrame(rows, columns=["customer_idx", "fold"]).assign(ap=rng.random(len(rows)) * 0.1)
    b = a.assign(ap=a.ap * 1.2)
    r = cluster_bootstrap(a, b, n_boot=200)
    assert r["n_customers"] == a.customer_idx.nunique() and r["n_rows"] == len(a)
    assert r["relative"] == pytest.approx(0.2)
    lo, hi = r["relative_ci95"]
    assert lo == pytest.approx(0.2) and hi == pytest.approx(0.2)   # an exact 20% gain has no ratio uncertainty
    same = cluster_bootstrap(a, a, n_boot=50)
    assert same["diff"] == 0 and same["diff_ci95"] == [0.0, 0.0]
    with pytest.raises(ValueError):
        cluster_bootstrap(a, b.iloc[:-1], n_boot=10)


# ---------------------------------------------------------------------------
# Ensemble: deriving a configuration from the superset equals building it directly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("train", [False, True])
def test_derivation_matches_direct_build(monkeypatch, train):
    import test_features as TF

    from ensemble.features import vocab as V
    from ensemble.features.track_a import build_features
    from ensemble.research.ensemble import channels_in, derive
    monkeypatch.setattr(V, "load_vocab", lambda con, cfg=None: V.build_vocab(con))
    # Superset: channel a at cap 3, channel b at cap 2. Configuration: a at cap 1, b at cap 2.
    sup_rows = [(0, 1, 0.9, 1, None, None), (0, 3, 0.5, 2, 0.8, 1), (0, 2, 0.4, 3, None, None),
                (1, 2, None, None, 0.7, 1), (1, 1, 0.3, 3, 0.6, 2), (1, 3, 0.2, 2, None, None)]

    def features(rows):
        con = TF.make_db(99)
        con.execute("DROP TABLE cand")
        con.execute("CREATE TEMP TABLE cand (customer_idx INT, article_id INT, a_score DOUBLE, a_rank BIGINT, "
                    "b_score DOUBLE, b_rank BIGINT)")
        con.executemany("INSERT INTO cand VALUES (?, ?, ?, ?, ?, ?)", rows)
        return build_features(con, TF.WEEK, with_labels=True)

    sup = features(sup_rows)
    spec = {"caps": {"a": 1, "b": 2}, "neural_features": []}
    got = derive(sup, spec, channels_in(sup.columns), train).reset_index(drop=True)
    direct_rows = [(c, a, s if r is not None and r <= 1 else None, r if r is not None and r <= 1 else None, bs, br)
                   for c, a, s, r, bs, br in sup_rows if (r is not None and r <= 1) or br is not None]
    want = features(direct_rows)
    if train:   # positives-only: keep customers with a purchased candidate in *this* configuration
        want = want[want.groupby("customer_idx")["label"].transform("max") > 0]
    pd.testing.assert_frame_equal(got, want.reset_index(drop=True), check_dtype=False)
    assert len(got) < len(sup)

"""Train one neural baseline for one label week and cache its outputs (own process).

  ENSEMBLE_CONFIG=track_a_research python -m ensemble.research.neural <bpr|lightgcn|sasrec> <week_start> <seed> ['{"json": "overrides"}'] [--curve]

Inputs stop at ``week.start`` (``ensemble.research.data``). Outputs, under
``data/interim/track_a_research/neural/<model>/<key>/`` where ``key`` hashes the model
parameters, the protocol version, the week and the seed:

- ``topk.parquet``  customer_idx, rank, article_id, score: exact top ``research.top_k`` over
  the eligible catalogue for every label-week buyer the model can represent;
- ``emb.npz``       query vectors of those buyers and vectors of every vocabulary article, so
  the ranker can score any candidate (dot product) without PyTorch;
- ``result.json``   parameters, data sizes, per-epoch log (loss; with ``--curve`` also MAP@12
  of the raw list on the label week, allowed on tuning folds only), wall time, peak RSS,
  parameter count, artifact bytes.

PyTorch and LightGBM bundle clashing OpenMP runtimes on macOS, so the ranker process never
imports this module; it reads the cached files.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble.config import load_config
from ensemble.db import connect
from ensemble.research import data as D
from ensemble.research.protocol import eligible, proto, top_k_from_scores, truth, tuning_folds, week_of

MODELS = ("bpr", "lightgcn", "sasrec")


def params_for(cfg, model: str, overrides: dict | None = None) -> dict:
    return {**dict(cfg.research.models[model]), **(overrides or {})}


def cache_dir(cfg, model: str, week_start: str, seed: int, params: dict) -> Path:
    body = json.dumps({"model": model, "week": week_start, "seed": int(seed), "params": params,
                       "protocol_version": int(proto(cfg).protocol_version),
                       "eligible_days": int(proto(cfg).eligible_days), "top_k": int(cfg.research.top_k)},
                      sort_keys=True, default=str)
    key = hashlib.sha1(body.encode()).hexdigest()[:12]
    return cfg.path("interim") / "track_a_research" / "neural" / model / f"{week_start}_s{seed}_{key}"


def peak_rss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


def _ap(pred, tr, k=12):
    hits, s = 0, 0.0
    for n, a in enumerate(pred[:k]):
        if a in tr:
            hits += 1
            s += hits / (n + 1)
    return s / min(len(tr), k)


def raw_map(users: np.ndarray, items: np.ndarray, tr: dict) -> float:
    """MAP@12 of the raw lists over all label-week buyers (no back-fill): a learning-curve signal only."""
    pos = {int(u): r for r, u in enumerate(users)}
    return float(np.mean([_ap(items[pos[u]].tolist(), t) if u in pos else 0.0 for u, t in tr.items()]))


def score_topk(Q: np.ndarray, I: np.ndarray, item_ids: np.ndarray, k: int, mask: list[np.ndarray] | None = None,
               block: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k of Q @ I.T, row blocks; ``mask[r]`` lists column indices to exclude for row r."""
    out_i, out_s = [], []
    for b in range(0, len(Q), block):
        S = Q[b:b + block] @ I.T
        if mask is not None:
            for r in range(len(S)):
                cols = mask[b + r]
                if len(cols):
                    S[r, cols] = -np.inf
        it, sc = top_k_from_scores(np.arange(len(Q[b:b + block])), item_ids, S, k)
        out_i.append(it)
        out_s.append(sc)
    if not out_i:
        return np.zeros((0, k), np.int64), np.zeros((0, k), np.float32)
    return np.vstack(out_i), np.vstack(out_s)


def run(model: str, week_start: str, seed: int, overrides: dict | None = None, curve: bool = False) -> Path:
    import torch

    from ensemble.research import models as M

    cfg = load_config()
    p = params_for(cfg, model, overrides)
    out = cache_dir(cfg, model, week_start, seed, p)
    if (out / "result.json").exists():
        return out
    week = week_of(week_start)
    if curve and week not in tuning_folds(cfg):
        raise SystemExit("learning curves read the label week: allowed on tuning folds only")
    torch.set_num_threads(int(p.get("threads", 8)))
    dev = M.device_of(p.get("device", "cpu"))
    M.seed_everything(seed)
    t_start = time.time()
    con = connect(cfg, read_only=True)
    targets = D.target_users(con, week)
    elig = eligible(con, week, int(proto(cfg).eligible_days))
    tr = truth(con, week) if curve else None
    k = int(cfg.research.top_k)
    info: dict = {"model": model, "week": week_start, "seed": seed, "params": p, "device": str(dev)}

    if model in ("bpr", "lightgcn"):
        X = D.interactions(con, week, int(p["window_weeks"]), int(p["min_item_count"]))
        con.close()
        nu, ni = len(X["users"]), len(X["items"])
        info.update(n_users=nu, n_items=ni, n_interactions=int(len(X["u"])))
        if model == "bpr":
            net = M.BPRMF(nu, ni, int(p["dim"])).to(dev)
        else:
            R, Rt = M.normalized_graph(X["u"], X["i"], nu, ni, dev)
            net = M.LightGCN(nu, ni, int(p["dim"]), int(p["layers"]), R, Rt).to(dev)
        upos = np.searchsorted(X["users"], targets)
        has = (upos < nu) & (X["users"][np.minimum(upos, nu - 1)] == targets)
        t_users, t_rows = targets[has], upos[has]
        cols = np.flatnonzero(np.isin(X["items"], elig))
        starts = np.searchsorted(X["u"], np.arange(nu + 1))

        def lists(net, mask_seen=bool(p.get("mask_seen"))):
            U, I = (x.float().cpu().numpy() for x in net.final())
            mask = None
            if mask_seen:
                col_of = np.full(ni, -1)
                col_of[cols] = np.arange(len(cols))
                mask = [col_of[X["i"][starts[r]:starts[r + 1]]] for r in t_rows]
                mask = [m[m >= 0] for m in mask]
            items, scores = score_topk(U[t_rows], I[cols], X["items"][cols], k, mask)
            return U, I, items, scores

        def on_epoch(ep, net):
            if curve and (ep % int(p.get("curve_every", 2)) == 0 or ep == int(p["epochs"])):
                return {"raw_map@12": raw_map(t_users, lists(net, False)[2], tr),
                        "raw_map@12_mask_seen": raw_map(t_users, lists(net, True)[2], tr)}
            return None

        t0 = time.time()
        log = M.train_pairwise(net, X["u"], X["i"], ni, p, seed, dev, on_epoch)
        train_s = time.time() - t0
        t0 = time.time()
        U, I, items, scores = lists(net)
        score_s = time.time() - t0
        q_users, Q, item_ids, IV = t_users, U[t_rows], X["items"], I
    elif model == "sasrec":
        vocab = D.vocabulary(con, week, int(p["vocab_weeks"]), int(p["min_item_count"]))
        S = D.sequences(con, week, vocab, int(p["history_weeks"]), int(p["max_len"]) + 1)
        con.close()
        V = len(vocab)
        train_mask = (S["last_day"] < 7 * int(p["train_weeks"])) & ((S["seq"] > 0).sum(1) >= 2)
        train_seq = S["seq"][train_mask]
        info.update(n_items=V, n_sequences=int(len(S["users"])), n_train_sequences=int(len(train_seq)),
                    n_train_tokens=int((train_seq[:, 1:] > 0).sum()))
        net = M.SASRec(V, int(p["max_len"]), int(p["dim"]), int(p["blocks"]), int(p["heads"]),
                       float(p["dropout"])).to(dev)
        upos = np.searchsorted(S["users"], targets)
        has = (upos < len(S["users"])) & (S["users"][np.minimum(upos, len(S["users"]) - 1)] == targets)
        t_users, t_rows = targets[has], upos[has]
        cols = np.flatnonzero(np.isin(vocab, elig))

        def query(net):
            net.eval()
            qs = []
            for b in range(0, len(t_rows), 2048):
                x = torch.from_numpy(S["seq"][t_rows[b:b + 2048], 1:].astype(np.int64)).to(dev)
                qs.append(net.represent(x).float().cpu().numpy())
            return np.vstack(qs) if qs else np.zeros((0, int(p["dim"])), np.float32)

        def lists(net, mask_seen=bool(p.get("mask_seen"))):
            Q = query(net)
            I = net.item.weight[1:].detach().float().cpu().numpy()
            mask = None
            if mask_seen:
                col_of = np.full(V, -1)
                col_of[cols] = np.arange(len(cols))
                mask = []
                for r in t_rows:
                    toks = S["seq"][r][S["seq"][r] > 0] - 1
                    m = col_of[toks]
                    mask.append(np.unique(m[m >= 0]))
            items, scores = score_topk(Q, I[cols], vocab[cols], k, mask)
            return Q, I, items, scores

        def on_epoch(ep, net):
            if curve and (ep % int(p.get("curve_every", 1)) == 0 or ep == int(p["epochs"])):
                return {"raw_map@12": raw_map(t_users, lists(net, False)[2], tr),
                        "raw_map@12_mask_seen": raw_map(t_users, lists(net, True)[2], tr)}
            return None

        t0 = time.time()
        log = M.train_sasrec(net, train_seq, V, p, seed, dev, on_epoch)
        train_s = time.time() - t0
        t0 = time.time()
        Q, IV, items, scores = lists(net)
        score_s = time.time() - t0
        q_users, item_ids = t_users, vocab
    else:
        raise SystemExit(f"unknown model {model}")

    out.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    n = len(q_users)
    pd.DataFrame({"customer_idx": np.repeat(q_users, items.shape[1]).astype(np.int32),
                  "rank": np.tile(np.arange(1, items.shape[1] + 1), n).astype(np.int16),
                  "article_id": items.ravel().astype(np.int32),
                  "score": scores.ravel().astype(np.float32)}).to_parquet(tmp / "topk.parquet", index=False)
    np.savez(tmp / "emb.npz", users=q_users.astype(np.int64), user_emb=Q.astype(np.float32),
             items=np.asarray(item_ids, np.int64), item_emb=np.asarray(IV, np.float32))
    for f in tmp.iterdir():
        f.replace(out / f.name)
    tmp.rmdir()
    info.update({"n_targets": int(len(targets)), "n_targets_represented": int(n),
                 "eligible_items": int(len(elig)), "eligible_scored": int(len(cols)),
                 "epochs_log": log, "train_seconds": round(train_s, 1), "score_seconds": round(score_s, 1),
                 "wall_seconds": round(time.time() - t_start, 1), "peak_rss_bytes": peak_rss_bytes(),
                 "mps_driver_bytes": int(torch.mps.driver_allocated_memory()) if dev.type == "mps" else 0,
                 "n_parameters": M.n_parameters(net),
                 "artifact_bytes": int(sum(f.stat().st_size for f in out.iterdir())),
                 "torch": torch.__version__})
    (out / "result.json").write_text(json.dumps(info, indent=1, default=float))
    return out


def run_subprocess(model: str, week_start: str, seed: int, overrides: dict | None = None, curve: bool = False,
                   log_path: Path | None = None) -> Path:
    """Run ``run`` in a fresh interpreter (OpenMP isolation from LightGBM) unless already cached."""
    import subprocess
    cfg = load_config()
    out = cache_dir(cfg, model, week_start, seed, params_for(cfg, model, overrides))
    if (out / "result.json").exists():
        return out
    cmd = [sys.executable, "-m", "ensemble.research.neural", model, week_start, str(seed), json.dumps(overrides or {})]
    if curve:
        cmd.append("--curve")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    with open(log_path or os.devnull, "a") as fh:
        r = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"{model} {week_start} seed {seed} failed with exit code {r.returncode} (log: {log_path})")
    return out


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--curve"]
    path = run(a[0], a[1], int(a[2]), json.loads(a[3]) if len(a) > 3 else None, "--curve" in sys.argv)
    print(path)

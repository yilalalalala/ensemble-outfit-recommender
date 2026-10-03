"""Minimal, faithful PyTorch reproductions of three published recommender baselines.

- **BPR-MF** (Rendle et al., UAI 2009): user/item latent vectors, score = dot product,
  pairwise BPR loss ``-log sigma(x_ui - x_uj)`` with L2 regularisation of the batch
  embeddings; negatives uniform over the training item vocabulary, rejecting the
  customer's own training positives.
- **LightGCN** (He et al., SIGIR 2020): ID embeddings propagated over the symmetric
  normalised user-item graph without feature transforms or non-linearities; the final
  representation is the mean of the layer-0..L embeddings; BPR loss with L2 on the layer-0
  (ego) embeddings of the batch; init N(0, 0.1) as in the authors' PyTorch code.
- **SASRec** (Kang & McAuley, ICDM 2018): causal self-attention over the last ``max_len``
  items, learned positions, item embeddings tied with the output layer, block structure of
  the widely used PyTorch port (pmixer/SASRec.pytorch). Objectives: ``bce`` = the original
  one-negative binary cross-entropy; ``gbce`` = gSASRec's generalised BCE with k sampled
  negatives (Petrov & Macdonald, RecSys 2023); ``ce`` = full-softmax cross-entropy over the
  vocabulary (Klenitskiy & Vasilev, RecSys 2023). Full CE costs ~9 min per epoch here
  (41k-item vocabulary, 740k sequences), so it is kept for reference and small runs.

Nothing here touches the database: inputs are the arrays from ``ensemble.research.data``.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def device_of(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(dev: torch.device) -> None:
    if dev.type == "mps":
        torch.mps.synchronize()


def empty_cache(dev: torch.device) -> None:
    if dev.type == "mps":
        torch.mps.empty_cache()


class NegativeSampler:
    """Uniform negatives over ``n_items``, re-drawn while they hit the customer's own positives."""

    def __init__(self, u: np.ndarray, i: np.ndarray, n_items: int, seed: int):
        self.n_items = int(n_items)
        self.keys = np.unique(u.astype(np.int64) * self.n_items + i.astype(np.int64))
        self.rng = np.random.default_rng(seed)

    def is_positive(self, u: np.ndarray, j: np.ndarray) -> np.ndarray:
        k = u.astype(np.int64) * self.n_items + j
        pos = np.minimum(np.searchsorted(self.keys, k), len(self.keys) - 1)
        return self.keys[pos] == k

    def sample(self, u: np.ndarray) -> np.ndarray:
        j = self.rng.integers(0, self.n_items, len(u))
        for _ in range(50):
            bad = self.is_positive(u, j)
            if not bad.any():
                break
            j[bad] = self.rng.integers(0, self.n_items, int(bad.sum()))
        return j


def bpr_loss(xu, xi, xj, reg_terms, reg: float) -> torch.Tensor:
    x = (xu * xi).sum(1) - (xu * xj).sum(1)
    l2 = sum(t.pow(2).sum() for t in reg_terms) / (2 * len(xu))
    return -F.logsigmoid(x).mean() + reg * l2


# ---------------------------------------------------------------------------
# BPR-MF
# ---------------------------------------------------------------------------

class BPRMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int):
        super().__init__()
        self.user = nn.Embedding(n_users, dim, sparse=True)
        self.item = nn.Embedding(n_items, dim, sparse=True)
        nn.init.normal_(self.user.weight, std=0.1)
        nn.init.normal_(self.item.weight, std=0.1)

    def loss(self, u, i, j, reg):
        eu, ei, ej = self.user(u), self.item(i), self.item(j)
        return bpr_loss(eu, ei, ej, (eu, ei, ej), reg)

    @torch.no_grad()
    def final(self):
        return self.user.weight.detach(), self.item.weight.detach()


# ---------------------------------------------------------------------------
# LightGCN
# ---------------------------------------------------------------------------

def normalized_graph(u: np.ndarray, i: np.ndarray, n_users: int, n_items: int, dev: torch.device):
    """R_hat = D_u^-1/2 R D_i^-1/2 (user x item) and its transpose: the two off-diagonal blocks of the
    symmetric normalised adjacency of the bipartite graph."""
    du = np.bincount(u, minlength=n_users).astype(np.float64)
    di = np.bincount(i, minlength=n_items).astype(np.float64)
    val = (1.0 / np.sqrt(du[u] * di[i])).astype(np.float32)
    idx = torch.from_numpy(np.vstack([u, i]).astype(np.int64))
    R = torch.sparse_coo_tensor(idx, torch.from_numpy(val), (n_users, n_items)).coalesce()
    Rt = torch.sparse_coo_tensor(idx.flip(0), torch.from_numpy(val), (n_items, n_users)).coalesce()
    return R.to(dev), Rt.to(dev)


class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int, layers: int, R, Rt):
        super().__init__()
        self.user = nn.Parameter(torch.empty(n_users, dim))
        self.item = nn.Parameter(torch.empty(n_items, dim))
        nn.init.normal_(self.user, std=0.1)
        nn.init.normal_(self.item, std=0.1)
        self.layers, self.R, self.Rt = int(layers), R, Rt

    def propagate(self):
        u, i = self.user, self.item
        su, si = u, i
        for _ in range(self.layers):
            u, i = torch.sparse.mm(self.R, i), torch.sparse.mm(self.Rt, u)
            su, si = su + u, si + i
        return su / (self.layers + 1), si / (self.layers + 1)

    def loss(self, u, i, j, reg):
        U, I = self.propagate()
        return bpr_loss(U[u], I[i], I[j], (self.user[u], self.item[i], self.item[j]), reg)

    @torch.no_grad()
    def final(self):
        U, I = self.propagate()
        return U.detach(), I.detach()


def train_pairwise(model, u: np.ndarray, i: np.ndarray, n_items: int, p: dict, seed: int, dev: torch.device,
                   on_epoch=None) -> list[dict]:
    """Shared BPR training loop for BPR-MF and LightGCN. Returns the per-epoch log."""
    sampler = NegativeSampler(u, i, n_items, seed + 1)
    rng = np.random.default_rng(seed + 2)
    if isinstance(model, BPRMF):
        opt = torch.optim.SparseAdam(list(model.parameters()), lr=float(p["lr"]))
    else:
        opt = torch.optim.Adam(model.parameters(), lr=float(p["lr"]))
    bs, log = int(p["batch_size"]), []
    for ep in range(1, int(p["epochs"]) + 1):
        t0, tot, nb = time.time(), 0.0, 0
        order = rng.permutation(len(u))
        for b in range(0, len(order), bs):
            sel = order[b:b + bs]
            ub, ib = u[sel], i[sel]
            jb = sampler.sample(ub)
            tu, ti, tj = (torch.from_numpy(x.astype(np.int64)).to(dev) for x in (ub, ib, jb))
            opt.zero_grad(set_to_none=True)
            loss = model.loss(tu, ti, tj, float(p["reg"]))
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        sync(dev)
        empty_cache(dev)
        rec = {"epoch": ep, "loss": tot / max(nb, 1), "seconds": round(time.time() - t0, 2)}
        if on_epoch is not None:
            rec.update(on_epoch(ep, model) or {})
        log.append(rec)
        print(f"    epoch {rec}", flush=True)
    return log


# ---------------------------------------------------------------------------
# SASRec
# ---------------------------------------------------------------------------

class PointWiseFF(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.l1, self.l2 = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.d1, self.d2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, x):
        return x + self.d2(self.l2(F.relu(self.d1(self.l1(x)))))


class SASRec(nn.Module):
    def __init__(self, n_items: int, max_len: int, dim: int, blocks: int, heads: int, dropout: float):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pos = nn.Embedding(max_len, dim)
        self.drop = nn.Dropout(dropout)
        self.attn_ln = nn.ModuleList(nn.LayerNorm(dim, eps=1e-8) for _ in range(blocks))
        self.attn = nn.ModuleList(nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
                                  for _ in range(blocks))
        self.ffn_ln = nn.ModuleList(nn.LayerNorm(dim, eps=1e-8) for _ in range(blocks))
        self.ffn = nn.ModuleList(PointWiseFF(dim, dropout) for _ in range(blocks))
        self.last_ln = nn.LayerNorm(dim, eps=1e-8)
        self.dim, self.max_len = dim, max_len
        for name, prm in self.named_parameters():
            if prm.dim() > 1:
                nn.init.xavier_normal_(prm)
        with torch.no_grad():
            self.item.weight[0].zero_()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        L = seq.shape[1]
        x = self.item(seq) * math.sqrt(self.dim)
        x = x + self.pos(torch.arange(L, device=seq.device))[None]
        x = self.drop(x)
        keep = (seq > 0).unsqueeze(-1).to(x.dtype)
        x = x * keep
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=seq.device), 1)
        for b in range(len(self.attn)):
            q = self.attn_ln[b](x)
            out, _ = self.attn[b](q, x, x, attn_mask=causal, need_weights=False)
            x = q + out
            x = self.ffn[b](self.ffn_ln[b](x))
            x = x * keep
        return self.last_ln(x)

    def loss(self, inp: torch.Tensor, tgt: torch.Tensor, kind: str, n_items: int, gen: torch.Generator | None = None,
             negatives: int = 256, gbce_t: float = 0.75):
        h = self(inp)
        m = tgt > 0
        if kind == "gbce":
            # gSASRec (Petrov & Macdonald, RecSys 2023): k uniform negatives per sequence, shared by its
            # positions; the positive term is raised to beta to undo the overconfidence that negative
            # sampling induces in BCE. alpha = k / (|I| - 1), beta = alpha * (t * (1 - 1/alpha) + 1/alpha).
            B = inp.shape[0]
            neg = torch.randint(1, n_items + 1, (B, negatives), device=inp.device, generator=gen)
            pos_l = (h * self.item(tgt)).sum(-1)[m]
            neg_l = torch.bmm(h, self.item(neg).transpose(1, 2))[m]
            alpha = negatives / (n_items - 1)
            beta = alpha * (gbce_t * (1 - 1 / alpha) + 1 / alpha)
            return -(beta * F.logsigmoid(pos_l)).mean() - F.logsigmoid(-neg_l).sum(-1).mean()
        hs, ts = h[m], tgt[m]
        if kind == "ce":
            logits = hs @ self.item.weight[1:].T
            return F.cross_entropy(logits, ts - 1)
        # Original SASRec: one uniform negative per position, binary cross-entropy.
        neg = torch.randint(1, n_items + 1, ts.shape, device=ts.device, generator=gen)
        pos_l = (hs * self.item(ts)).sum(-1)
        neg_l = (hs * self.item(neg)).sum(-1)
        return F.binary_cross_entropy_with_logits(pos_l, torch.ones_like(pos_l)) + \
            F.binary_cross_entropy_with_logits(neg_l, torch.zeros_like(neg_l))

    @torch.no_grad()
    def represent(self, seq: torch.Tensor) -> torch.Tensor:
        """Hidden state at the last position: the customer's next-item query vector."""
        return self(seq)[:, -1, :]


def train_sasrec(model: SASRec, seq: np.ndarray, n_items: int, p: dict, seed: int, dev: torch.device,
                 on_epoch=None) -> list[dict]:
    """``seq`` holds max_len + 1 tokens per training customer: inputs are [:, :-1], targets [:, 1:]."""
    rng = np.random.default_rng(seed + 2)
    gen = torch.Generator(device=dev.type).manual_seed(seed + 3) if p.get("loss") in ("bce", "gbce") else None
    opt = torch.optim.Adam(model.parameters(), lr=float(p["lr"]), betas=(0.9, 0.98))
    bs, log = int(p["batch_size"]), []
    for ep in range(1, int(p["epochs"]) + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        order = rng.permutation(len(seq))
        for b in range(0, len(order), bs):
            batch = torch.from_numpy(seq[order[b:b + bs]].astype(np.int64)).to(dev)
            opt.zero_grad(set_to_none=True)
            loss = model.loss(batch[:, :-1], batch[:, 1:], p.get("loss", "gbce"), n_items, gen,
                              int(p.get("negatives", 256)), float(p.get("gbce_t", 0.75)))
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        sync(dev)
        empty_cache(dev)
        model.eval()
        rec = {"epoch": ep, "loss": tot / max(nb, 1), "seconds": round(time.time() - t0, 2)}
        if on_epoch is not None:
            rec.update(on_epoch(ep, model) or {})
        log.append(rec)
        print(f"    epoch {rec}", flush=True)
    return log


def n_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))

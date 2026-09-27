"""Track B recommenders: popularity baseline, association rules, two-tower, and fusion."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- baselines


def pop_buckets(uni: pd.DataFrame, n: int = 10) -> dict[int, int]:
    out = {}
    for _, g in uni.groupby("slot"):
        sold = g[g.pop_window > 0]
        ranks = sold.pop_window.rank(method="first", pct=True)
        out.update(dict(zip(sold.article_id, np.ceil(ranks * n).astype(int))))
    return out


def slot_lists(uni: pd.DataFrame) -> dict[str, np.ndarray]:
    """Universe per slot, ordered by recent popularity (the popularity baseline ranking)."""
    return {s: g.article_id.values for s, g in uni.groupby("slot", sort=False)}


def popularity(q: pd.DataFrame, uni: pd.DataFrame, k: int) -> list[np.ndarray]:
    lists = slot_lists(uni)
    return [lists[s][:k] for s in q.target_slot.values]


def association(con, q: pd.DataFrame, uni: pd.DataFrame, k: int, min_support: int, score: str = "npmi") -> list[np.ndarray]:
    """Market basket analysis: rank the anchor's co-purchased items in the target slot.

    ``score='npmi'`` ranks by normalised PMI (controls for popularity); ``score='co'``
    ranks by raw co-occurrence count (the "PMI off" ablation). Items with no
    qualifying pair are back-filled by popularity.
    """
    live = set(uni.article_id)
    order = "npmi DESC" if score == "npmi" else "co DESC"
    cond = "AND lift > 1" if score == "npmi" else ""
    rows = con.execute(f"""
        SELECT src, dst_slot, dst FROM pairs_all
        WHERE co >= {int(min_support)} {cond} AND src IN (SELECT DISTINCT anchor FROM _q_anchors)
        QUALIFY row_number() OVER (PARTITION BY src, dst_slot ORDER BY {order}, dst) <= {4 * k}
        ORDER BY src, dst_slot, {order}, dst""").fetchall()
    nbrs: dict[tuple, list[int]] = defaultdict(list)
    for src, slot, dst in rows:
        if dst in live:
            nbrs[(src, slot)].append(dst)
    pop = slot_lists(uni)
    out = []
    for a, s in zip(q.anchor.values, q.target_slot.values):
        head = nbrs.get((a, s), [])[:k]
        if len(head) < k:
            seen = set(head)
            head = head + [x for x in pop[s][: 2 * k] if x not in seen][: k - len(head)]
        out.append(np.asarray(head))
    return out


def rrf(lists: list[list[np.ndarray]], weights: list[float], k: int, c: float = 60.0) -> list[np.ndarray]:
    """Weighted reciprocal rank fusion, a standard way to merge ranked lists."""
    out = []
    for per_query in zip(*lists):
        score: dict[int, float] = defaultdict(float)
        for w, lst in zip(weights, per_query):
            for r, a in enumerate(lst):
                score[a] += w / (c + r + 1)
        out.append(np.asarray(sorted(score, key=lambda a: -score[a])[:k]))
    return out


# ---------------------------------------------------------------- two-tower

FEATURES = ["product_type_no", "product_group_name", "graphical_appearance_no", "colour_group_code",
            "perceived_colour_value_id", "perceived_colour_master_id", "department_no", "index_code",
            "index_group_no", "section_no", "garment_group_no", "price_tier", "pop_bucket", "slot"]
SLOT_IDS = {s: i for i, s in enumerate(["upper", "lower", "full", "shoes", "accessories", "socks", "swimwear"])}


class ItemEncoder:
    """Maps article ids to integer feature codes (index 0 = unknown)."""

    def __init__(self, items: pd.DataFrame, tiers: dict[int, int], id_articles: np.ndarray,
                 pop_buckets: dict[int, int] | None = None):
        items = items.copy()
        items["price_tier"] = items.article_id.map(tiers).fillna(0).astype(int)
        # Recent-sales decile within the slot (0 = no sales before the cutoff), so the
        # towers can learn how much popularity matters instead of inferring it.
        items["pop_bucket"] = items.article_id.map(pop_buckets or {}).fillna(0).astype(int)
        self.row = {a: i for i, a in enumerate(items.article_id.values)}
        codes = []
        self.cardinality = []
        for f in FEATURES:
            vals = items[f].astype(object).where(items[f].notna(), "NA").astype(str)
            vocab = {v: i + 1 for i, v in enumerate(sorted(vals.unique()))}
            codes.append(vals.map(vocab).values)
            self.cardinality.append(len(vocab) + 1)
        id_map = {a: i + 1 for i, a in enumerate(id_articles)}
        codes.append(items.article_id.map(id_map).fillna(0).astype(int).values)
        self.cardinality.append(len(id_articles) + 1)
        self.codes = torch.as_tensor(np.stack(codes, axis=1), dtype=torch.long)

    def __call__(self, article_ids) -> torch.Tensor:
        return self.codes[[self.row[a] for a in article_ids]]


class Tower(nn.Module):
    def __init__(self, cardinality: list[int], emb_dim: int, hidden: int, dim: int, extra: int = 0, id_dim: int = 32):
        super().__init__()
        self.embs = nn.ModuleList(nn.Embedding(n, emb_dim) for n in cardinality[:-1])
        self.id_emb = nn.Embedding(cardinality[-1], id_dim, padding_idx=0)
        self.extra = nn.Embedding(extra, emb_dim) if extra else None
        d_in = emb_dim * (len(cardinality) - 1) + id_dim + (emb_dim if extra else 0)
        self.mlp = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def forward(self, codes: torch.Tensor, id_keep: torch.Tensor | None = None, extra: torch.Tensor | None = None):
        parts = [e(codes[:, i]) for i, e in enumerate(self.embs)]
        ids = self.id_emb(codes[:, -1])
        if id_keep is not None:
            ids = ids * id_keep.unsqueeze(1)
        parts.append(ids)
        if self.extra is not None:
            parts.append(self.extra(extra))
        return F.normalize(self.mlp(torch.cat(parts, dim=1)), dim=1)


class TwoTower(nn.Module):
    """Anchor tower (conditioned on the target slot) and complement tower."""

    def __init__(self, cardinality, tt):
        super().__init__()
        self.query = Tower(cardinality, int(tt.emb_dim), int(tt.hidden), int(tt.dim), extra=len(SLOT_IDS))
        self.item = Tower(cardinality, int(tt.emb_dim), int(tt.hidden), int(tt.dim))
        self.tau = float(tt.temperature)


def train_two_tower(pos: pd.DataFrame, enc: ItemEncoder, uni_items: pd.DataFrame, tiers: dict, tt,
                    device: str = "cpu", seed: int = 0, log=print) -> TwoTower:
    """Sampled-softmax training over in-batch plus sampled negatives.

    ``pos`` has one row per training example (src, dst, dst_slot). Negative
    options (``tt.negatives``):
      in_batch  other positives in the batch with the same target slot
      logq      subtract log(sampling frequency) from in-batch/pop logits (Yi et al., 2019)
      pop       per-slot negatives drawn ∝ popularity^0.75
      hard      same product type and price tier as the positive
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    neg = set(tt.negatives)
    model = TwoTower(enc.cardinality, tt).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(tt.lr))

    src_codes = enc(pos.src.values)
    dst_codes = enc(pos.dst.values)
    slot_ids = torch.as_tensor(pos.dst_slot.map(SLOT_IDS).values, dtype=torch.long)
    # Sampling frequency of each item as a positive, for logQ correction.
    freq = pos.dst.value_counts()
    logq_dst = torch.as_tensor(np.log(pos.dst.map(freq).values / len(pos)), dtype=torch.float32)

    # Popularity-based negative pools per slot.
    pools = {}
    for s, g in uni_items.groupby("slot"):
        p = (g.pop_window.values + 1.0) ** 0.75
        pools[SLOT_IDS[s]] = (g.article_id.values, p / p.sum())
    # Hard negative groups: (product type, price tier).
    items = uni_items.assign(tier=uni_items.article_id.map(tiers).fillna(0).astype(int),
                             ptype=uni_items.article_id.map(dict(zip(enc_items_ids(enc), enc_types(enc)))))
    groups = {k: g.article_id.values for k, g in items.groupby(["ptype", "tier"])}
    key_of = dict(zip(items.article_id, zip(items.ptype, items.tier)))

    n = len(pos)
    bs = int(tt.batch_size)
    for epoch in range(int(tt.epochs)):
        perm = torch.as_tensor(rng.permutation(n))
        total, steps = 0.0, 0
        for b in range(0, n - bs + 1, bs):
            idx = perm[b:b + bs]
            s_codes, d_codes, sl = src_codes[idx], dst_codes[idx], slot_ids[idx]
            keep_q = (torch.rand(len(idx)) > float(tt.id_dropout)).float()
            keep_d = (torch.rand(len(idx)) > float(tt.id_dropout)).float()
            q = model.query(s_codes.to(device), keep_q.to(device), sl.to(device))
            d = model.item(d_codes.to(device), keep_d.to(device))
            logits = [q @ d.T / model.tau]
            if "logq" in neg:
                logits[0] = logits[0] - logq_dst[idx].to(device).unsqueeze(0)
            same_slot = (sl.unsqueeze(0) == sl.unsqueeze(1)).to(device)
            same_item = (d_codes[:, -1].unsqueeze(0) == d_codes[:, -1].unsqueeze(1)).to(device) & (d_codes[:, -1].unsqueeze(0) > 0).to(device)
            eye = torch.eye(len(idx), dtype=torch.bool, device=device)
            mask = ~same_slot | (same_item & ~eye)
            if "in_batch" not in neg:
                mask = ~eye
            logits[0] = logits[0].masked_fill(mask, -1e9)

            if "pop" in neg:
                m = int(tt.pop_negatives)
                pl = torch.full((len(idx), m), -1e9, device=device)
                for s_id in sl.unique().tolist():
                    arts, p = pools[s_id]
                    sample = rng.choice(len(arts), size=m, p=p)
                    e = model.item(enc(arts[sample]).to(device), torch.zeros(m, device=device))
                    rows = (sl == s_id).nonzero().squeeze(1).to(device)
                    sc = q[rows] @ e.T / model.tau
                    if "logq" in neg:
                        sc = sc - torch.as_tensor(np.log(p[sample] * m), dtype=torch.float32, device=device)
                    pl[rows] = sc
                logits.append(pl)
            if "hard" in neg:
                h = int(tt.hard_negatives)
                hard_ids = []
                for a in pos.dst.values[idx.numpy()]:
                    g = groups.get(key_of.get(a), None)
                    hard_ids.extend(rng.choice(g, size=h) if g is not None and len(g) > 1 else [a] * h)
                e = model.item(enc(hard_ids).to(device), torch.zeros(len(hard_ids), device=device)).view(len(idx), h, -1)
                hs = (q.unsqueeze(1) * e).sum(-1) / model.tau
                same = torch.as_tensor(np.asarray(hard_ids).reshape(len(idx), h) == pos.dst.values[idx.numpy()][:, None], device=device)
                logits.append(hs.masked_fill(same, -1e9))
            all_logits = torch.cat(logits, dim=1)
            loss = F.cross_entropy(all_logits, torch.arange(len(idx), device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        log(f"    epoch {epoch + 1}: loss {total / max(steps, 1):.4f}")
    return model


def enc_items_ids(enc: ItemEncoder):
    return list(enc.row.keys())


def enc_types(enc: ItemEncoder):
    return enc.codes[:, FEATURES.index("product_type_no")].tolist()


@torch.no_grad()
def two_tower_recs(model: TwoTower, enc: ItemEncoder, q: pd.DataFrame, uni: pd.DataFrame, k: int,
                   device: str = "cpu", use_ids: bool = True) -> list[np.ndarray]:
    model.eval()
    out: list[np.ndarray | None] = [None] * len(q)
    for s, g in uni.groupby("slot"):
        arts = g.article_id.values
        e = torch.cat([model.item(enc(arts[i:i + 8192]).to(device),
                                  torch.full((len(arts[i:i + 8192]),), float(use_ids), device=device))
                       for i in range(0, len(arts), 8192)])
        rows = np.flatnonzero(q.target_slot.values == s)
        for b in range(0, len(rows), 4096):
            r = rows[b:b + 4096]
            qe = model.query(enc(q.anchor.values[r]).to(device),
                             torch.full((len(r),), float(use_ids), device=device),
                             torch.full((len(r),), SLOT_IDS[s], dtype=torch.long, device=device))
            top = torch.topk(qe @ e.T, min(k, len(arts)), dim=1).indices.cpu().numpy()
            for i, t in zip(r, top):
                out[i] = arts[t]
    model.train()
    return out

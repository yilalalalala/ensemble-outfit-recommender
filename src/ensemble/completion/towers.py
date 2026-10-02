"""Track B two-tower complement retrieval, rebuilt for Round 3 (Phase 5).

Same architecture as the Round-1 model (``ensemble.completion.models.TwoTower``)
so the shipped baseline stays reproducible, but three things change.

**Speed.** The Round-1 trainer re-encoded every batch through a Python ``dict``
lookup (``{article_id: row}``) twice per step and drew hard negatives with one
``rng.choice`` call per positive — 2048 Python calls per batch. An epoch over
3.2 M pairs took ~370 s. Here article ids are mapped to rows once with
``np.searchsorted`` and every per-batch tensor is a gather on a precomputed
matrix, which is what makes a rolling backtest affordable.

**Negatives.** The Round-1 ablation showed logQ is essential and that
popularity-sampled and (product-type, price-tier) hard negatives added nothing.
Those are removed. In their place is *retrieval-informed* hard negative mining:
each step scores a popularity-sampled pool against the batch queries and takes
the highest-scoring wrong items, with likely false negatives (the positive
itself, its colourways, and articles with basket co-occurrence evidence for the
anchor) masked out instead of being learned against.

**Model selection.** The last ``holdout_days`` of the mining window are held out
as baskets; after each epoch the model is scored by Recall@12 on a sample of
held-out queries over the full slot catalogue, and the best epoch's weights are
restored. No target-week information is involved.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ensemble.completion.models import FEATURES, SLOT_IDS, Tower, TwoTower

PAIR_KEY_SHIFT = 1 << 32


def device_for(cfg) -> str:
    want = cfg.track_b.get("device", "auto")
    if want != "auto":
        return want
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Encoder:
    """Article → integer feature codes, with vectorised row lookup.

    ``codes`` is (n_articles, len(FEATURES) + 1); the last column is the article-ID
    embedding index (0 = held out / unknown, which is what ``id_dropout`` and the
    content-only evaluation exercise).
    """

    def __init__(self, items: pd.DataFrame, tiers: dict[int, int], id_articles: np.ndarray,
                 pop_bucket: dict[int, int] | None = None, clip: tuple[np.ndarray, np.ndarray] | None = None):
        items = items.sort_values("article_id").reset_index(drop=True)
        self.article_ids = items.article_id.to_numpy(dtype=np.int64)
        items = items.assign(price_tier=items.article_id.map(tiers).fillna(0).astype(int),
                             pop_bucket=items.article_id.map(pop_bucket or {}).fillna(0).astype(int))
        codes, self.cardinality, self.vocabs = [], [], {}
        for f in FEATURES:
            vals = items[f].astype(object).where(items[f].notna(), "NA").astype(str)
            vocab = {v: i + 1 for i, v in enumerate(sorted(vals.unique()))}
            self.vocabs[f] = vocab
            codes.append(vals.map(vocab).to_numpy())
            self.cardinality.append(len(vocab) + 1)
        id_map = {a: i + 1 for i, a in enumerate(np.asarray(id_articles))}
        codes.append(items.article_id.map(id_map).fillna(0).astype(int).to_numpy())
        self.cardinality.append(len(id_articles) + 1)
        self.codes = torch.as_tensor(np.stack(codes, axis=1).astype(np.int64))
        self.product_code = items.product_code.to_numpy()
        self.clip = None
        if clip is not None:
            ids, emb = clip
            pos = pd.Series(np.arange(len(ids)), index=np.asarray(ids, dtype=np.int64))
            idx = pos.reindex(self.article_ids).to_numpy()
            m = np.zeros((len(items), emb.shape[1]), dtype=np.float32)
            ok = ~np.isnan(idx)
            m[ok] = emb[idx[ok].astype(int)]
            self.clip = torch.as_tensor(m)

    def rows(self, article_ids) -> np.ndarray:
        a = np.asarray(article_ids, dtype=np.int64)
        i = np.searchsorted(self.article_ids, a)
        i = np.clip(i, 0, len(self.article_ids) - 1)
        if not np.all(self.article_ids[i] == a):
            raise KeyError("article id not in encoder")
        return i

    @property
    def clip_dim(self) -> int:
        return 0 if self.clip is None else int(self.clip.shape[1])


class Towers:
    """A trained :class:`TwoTower` plus the tensors needed to score it quickly."""

    def __init__(self, model: TwoTower, enc: Encoder, device: str):
        self.model, self.enc, self.device = model, enc, device
        self.codes = enc.codes.to(device)
        self.clip = None if enc.clip is None else enc.clip.to(device)

    def _tower(self, tower: Tower, rows: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        return tower(self.codes[rows], keep, clip=None if self.clip is None else self.clip[rows])

    def encode_items(self, rows: torch.Tensor, use_ids: bool) -> torch.Tensor:
        keep = torch.full((len(rows),), float(use_ids), device=self.device)
        return self._tower(self.model.item, rows, keep)

    def encode_queries(self, rows: torch.Tensor, slot_id: int, use_ids: bool) -> torch.Tensor:
        keep = torch.full((len(rows),), float(use_ids), device=self.device)
        slots = torch.full((len(rows),), slot_id, dtype=torch.long, device=self.device)
        c = None if self.clip is None else self.clip[rows]
        return self.model.query(self.codes[rows], keep, slots, c)

    @torch.no_grad()
    def retrieve(self, keys: pd.DataFrame, uni: pd.DataFrame, k: int, use_ids: bool = True) -> pd.DataFrame:
        """Top ``k`` eligible items per (anchor, target_slot) key, with cosine similarity."""
        self.model.eval()
        parts = []
        for slot, g in uni.groupby("slot", sort=False):
            arts = g.article_id.to_numpy(dtype=np.int64)
            rows = torch.as_tensor(self.enc.rows(arts), device=self.device)
            emb = torch.cat([self.encode_items(rows[i:i + 8192], use_ids) for i in range(0, len(rows), 8192)])
            kk = keys[keys.target_slot.values == slot]
            if not len(kk):
                continue
            anchors = kk.anchor.to_numpy(dtype=np.int64)
            arows = torch.as_tensor(self.enc.rows(anchors), device=self.device)
            for i in range(0, len(arows), 4096):
                qe = self.encode_queries(arows[i:i + 4096], SLOT_IDS[slot], use_ids)
                sim, top = torch.topk(qe @ emb.T, min(k, len(arts)), dim=1)
                n = sim.shape[0]
                parts.append(pd.DataFrame({
                    "anchor": np.repeat(anchors[i:i + 4096], sim.shape[1]),
                    "target_slot": slot,
                    "article_id": arts[top.cpu().numpy().ravel()],
                    "sim": sim.cpu().numpy().ravel().astype(np.float32)}))
                del qe, sim, top
            del emb, rows
        self.model.train()
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=["anchor", "target_slot", "article_id", "sim"])

    @torch.no_grad()
    def pair_similarity(self, anchors: np.ndarray, candidates: np.ndarray, slots: np.ndarray,
                        use_ids: bool = True, chunk: int = 200_000) -> np.ndarray:
        """Cosine similarity for arbitrary (anchor, candidate, target_slot) triples."""
        self.model.eval()
        out = np.zeros(len(anchors), dtype=np.float32)
        slot_ids = np.asarray([SLOT_IDS[s] for s in slots])
        for s_id in np.unique(slot_ids):
            idx = np.flatnonzero(slot_ids == s_id)
            for i in range(0, len(idx), chunk):
                j = idx[i:i + chunk]
                qr = torch.as_tensor(self.enc.rows(anchors[j]), device=self.device)
                cr = torch.as_tensor(self.enc.rows(candidates[j]), device=self.device)
                qe = self.encode_queries(qr, int(s_id), use_ids)
                ce = self.encode_items(cr, use_ids)
                out[j] = (qe * ce).sum(1).cpu().numpy()
        self.model.train()
        return out


def _pair_keys(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    return src.astype(np.int64) * PAIR_KEY_SHIFT + dst.astype(np.int64)


def train(con, pos: pd.DataFrame, enc: Encoder, uni: pd.DataFrame, cfg, holdout: pd.DataFrame | None = None,
          seed: int | None = None, negatives: list[str] | None = None, log=print) -> tuple[Towers, dict]:
    """Fit the towers on ``pos`` (columns src, dst, dst_slot) mined before the cutoff.

    ``holdout`` has the same columns and comes from the tail of the mining
    window; it is used only to pick the number of epochs.
    """
    tt = cfg.track_b.two_tower
    seed = int(tt.seed if seed is None else seed)
    neg = set(negatives if negatives is not None else tt.negatives)
    dev = device_for(cfg)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TwoTower(enc.cardinality, tt, enc.clip_dim).to(dev)
    towers = Towers(model, enc, dev)
    opt = torch.optim.Adam(model.parameters(), lr=float(tt.lr))

    src_rows = torch.as_tensor(enc.rows(pos.src.values), device=dev)
    dst_rows = torch.as_tensor(enc.rows(pos.dst.values), device=dev)
    slot_ids = torch.as_tensor(pos.dst_slot.map(SLOT_IDS).to_numpy(dtype=np.int64), device=dev)
    freq = pos.dst.value_counts()
    logq = torch.as_tensor(np.log(pos.dst.map(freq).to_numpy(dtype=np.float64) / len(pos)).astype(np.float32),
                           device=dev)
    code_of = torch.as_tensor(pd.factorize(enc.product_code)[0].astype(np.int64), device=dev)

    # Popularity-sampled pool per slot, re-scored every step so the hardest wrong
    # items are the ones the current model would retrieve (Phase 5.2).
    pool_rows, pool_p = {}, {}
    for s, g in uni.groupby("slot", sort=False):
        p = (g.pop_window.to_numpy(dtype=np.float64) + 1.0) ** 0.75
        pool_rows[SLOT_IDS[s]] = enc.rows(g.article_id.to_numpy(dtype=np.int64))
        pool_p[SLOT_IDS[s]] = p / p.sum()
    known_pairs = np.sort(con.execute("SELECT src, dst FROM tb_pairs").df().pipe(
        lambda d: _pair_keys(d.src.to_numpy(), d.dst.to_numpy())))

    n, bs = len(pos), int(tt.batch_size)
    n_hard, pool_n = int(tt.hard_negatives), int(tt.hard_pool)
    hist: list[dict] = []
    best = {"recall": -1.0, "epoch": 0, "state": None}
    patience = int(tt.early_stopping_patience)
    hv = _holdout_queries(holdout, rng) if holdout is not None and len(holdout) else None
    for epoch in range(int(tt.epochs)):
        t0 = time.time()
        perm = torch.as_tensor(rng.permutation(n), device=dev)
        total, steps = 0.0, 0
        for b in range(0, n - bs + 1, bs):
            idx = perm[b:b + bs]
            m = len(idx)
            sl = slot_ids[idx]
            keep_q = (torch.rand(m, device=dev) > float(tt.id_dropout)).float()
            keep_d = (torch.rand(m, device=dev) > float(tt.id_dropout)).float()
            qe = towers.model.query(towers.codes[src_rows[idx]], keep_q, sl,
                                    None if towers.clip is None else towers.clip[src_rows[idx]])
            de = towers._tower(towers.model.item, dst_rows[idx], keep_d)
            logits = qe @ de.T / model.tau
            if "logq" in neg:
                logits = logits - logq[idx].unsqueeze(0)
            eye = torch.eye(m, dtype=torch.bool, device=dev)
            if "in_batch" in neg:
                same_slot = sl.unsqueeze(0) == sl.unsqueeze(1)
                same_item = dst_rows[idx].unsqueeze(0) == dst_rows[idx].unsqueeze(1)
                logits = logits.masked_fill((~same_slot | same_item) & ~eye, -1e9)
            else:
                logits = logits.masked_fill(~eye, -1e9)
            parts = [logits]
            if n_hard and "hard" in neg:
                parts.append(_hard_logits(towers, qe, idx, pos, dst_rows, sl, pool_rows, pool_p, pool_n,
                                          n_hard, code_of, known_pairs, rng, dev, model.tau))
            loss = F.cross_entropy(torch.cat(parts, dim=1), torch.arange(m, device=dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        rec = _holdout_recall(towers, hv, uni, int(cfg.track_b.k)) if hv is not None else float("nan")
        hist.append({"epoch": epoch + 1, "loss": total / max(steps, 1), "holdout_recall@12": rec,
                     "seconds": round(time.time() - t0, 1)})
        log(f"    epoch {epoch + 1}: loss {hist[-1]['loss']:.4f} holdout R@12 "
            f"{rec:.4f} ({hist[-1]['seconds']:.0f}s)")
        if hv is not None:
            if rec > best["recall"]:
                best = {"recall": rec, "epoch": epoch + 1,
                        "state": copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})}
            elif epoch + 1 - best["epoch"] >= patience:
                break
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    info = {"epochs_run": len(hist), "best_epoch": best["epoch"], "best_holdout_recall@12": best["recall"],
            "history": hist, "negatives": sorted(neg), "hard_negatives": n_hard, "seed": seed, "device": dev}
    return towers, info


def _hard_logits(towers, qe, idx, pos, dst_rows, sl, pool_rows, pool_p, pool_n, n_hard,
                 code_of, known_pairs, rng, dev, tau) -> torch.Tensor:
    """Top-``n_hard`` wrong items from a popularity-sampled pool, false negatives masked."""
    m = len(idx)
    out = torch.full((m, n_hard), -1e9, device=dev)
    for s_id in torch.unique(sl).tolist():
        rowsel = (sl == s_id).nonzero().squeeze(1)
        arts, p = pool_rows[s_id], pool_p[s_id]
        take = rng.choice(len(arts), size=min(pool_n, len(arts)), replace=False, p=p / p.sum())
        prows = torch.as_tensor(arts[take], device=dev)
        pe = towers.encode_items(prows, use_ids=False)
        sc = qe[rowsel] @ pe.T / tau
        top = torch.topk(sc, min(n_hard, sc.shape[1]), dim=1)
        hard_rows = prows[top.indices]
        pos_rows = dst_rows[idx][rowsel].unsqueeze(1)
        bad = (hard_rows == pos_rows) | (code_of[hard_rows] == code_of[pos_rows])
        src_ids = pos.src.values[idx.cpu().numpy()][rowsel.cpu().numpy()]
        cand_ids = towers.enc.article_ids[hard_rows.cpu().numpy()]
        keys = _pair_keys(np.repeat(src_ids, cand_ids.shape[1]).reshape(cand_ids.shape), cand_ids)
        j = np.searchsorted(known_pairs, keys)
        j = np.clip(j, 0, len(known_pairs) - 1)
        co_known = known_pairs[j] == keys
        bad = bad | torch.as_tensor(co_known, device=dev)
        vals = top.values.masked_fill(bad, -1e9)
        out[rowsel, :vals.shape[1]] = vals
    return out


def _holdout_queries(holdout: pd.DataFrame, rng, n: int = 4000) -> pd.DataFrame:
    """Sampled (anchor, target_slot) keys with their held-out complements."""
    g = (holdout.drop_duplicates(["src", "dst_slot", "dst"])
         .groupby(["src", "dst_slot"], sort=True)["dst"].apply(list).reset_index())
    if len(g) > n:
        g = g.iloc[rng.choice(len(g), size=n, replace=False)].reset_index(drop=True)
    return g.rename(columns={"src": "anchor", "dst_slot": "target_slot", "dst": "truth"})


def _holdout_recall(towers: Towers, hv: pd.DataFrame, uni: pd.DataFrame, k: int) -> float:
    recs = towers.retrieve(hv[["anchor", "target_slot"]], uni, k)
    got = recs.groupby(["anchor", "target_slot"])["article_id"].agg(set).to_dict()
    hit = tot = 0
    for a, s, t in hv[["anchor", "target_slot", "truth"]].itertuples(index=False):
        p = got.get((a, s), set())
        hit += sum(1 for x in t if x in p)
        tot += len(t)
    return hit / tot if tot else float("nan")


# ------------------------------------------------------------------ per-week artifact process


@torch.no_grad()
def clip_similarity(enc: Encoder, anchors: np.ndarray, cands: np.ndarray, device: str,
                    chunk: int = 100_000) -> np.ndarray:
    """Cosine similarity of frozen FashionCLIP image vectors (0 when either article has no image)."""
    if enc.clip is None:
        return np.zeros(len(anchors), dtype=np.float32)
    emb = F.normalize(enc.clip, dim=1).to(device)
    ar = torch.as_tensor(enc.rows(anchors), device=device)
    cr = torch.as_tensor(enc.rows(cands), device=device)
    out = np.empty(len(anchors), dtype=np.float32)
    for i in range(0, len(anchors), chunk):
        out[i:i + chunk] = (emb[ar[i:i + chunk]] * emb[cr[i:i + chunk]]).sum(1).cpu().numpy()
    del emb, ar, cr
    return out


def build_artifacts(week_start: str, cfg=None, log=print) -> dict:
    """Train the towers for one week and cache retrieval lists plus pair similarities.

    Runs as its own process so PyTorch never shares an address space with
    LightGBM (``ensemble.completion.pipeline``).
    """
    import json
    from datetime import date

    from ensemble.completion import candidates as C
    from ensemble.completion import pipeline as PL
    from ensemble.completion.data import item_table
    from ensemble.completion.models import pop_buckets
    from ensemble.config import load_config
    from ensemble.data.splits import Week, load_splits
    from ensemble.db import connect

    cfg = cfg or load_config()
    con = connect(cfg, read_only=True)
    splits = load_splits(con, cfg)
    y, m, d = (int(x) for x in week_start.split("-"))
    week = Week(date(y, m, d), date(y, m, d) + (splits.val.end - splits.val.start))
    paths = PL.tower_paths(cfg, week)
    t0 = time.time()
    prep = PL.prepare_sql(con, cfg, week)
    uni, q = prep["uni"], prep["q"]
    tr, hold = PL.mining_pairs(con, int(cfg.track_b.two_tower.holdout_days))
    enc = Encoder(item_table(con), PL.price_tiers(con, week),
                  np.unique(np.r_[tr.src.to_numpy(), tr.dst.to_numpy()]), pop_buckets(uni),
                  clip=PL.load_clip(cfg))
    tw, info = train(con, tr, enc, uni, cfg, holdout=hold, log=log)
    keys = q[["anchor", "target_slot"]].drop_duplicates().reset_index(drop=True)
    cand = cfg.track_b.candidates
    tt = tw.retrieve(keys, uni, int(cand["two_tower"])) if int(cand["two_tower"]) else None
    ttc = (tw.retrieve(keys, uni, int(cand["two_tower_content"]), use_ids=False)
           if int(cand["two_tower_content"]) else None)
    C.build_sources(con, cfg, tt, ttc)
    key_cand = PL.key_candidates(con)
    key_cand["tt_pair_sim"] = tw.pair_similarity(key_cand.anchor.to_numpy(), key_cand.article_id.to_numpy(),
                                                 key_cand.target_slot.to_numpy(), use_ids=True)
    key_cand["ttc_pair_sim"] = tw.pair_similarity(key_cand.anchor.to_numpy(), key_cand.article_id.to_numpy(),
                                                  key_cand.target_slot.to_numpy(), use_ids=False)
    key_cand["clip_sim"] = clip_similarity(enc, key_cand.anchor.to_numpy(), key_cand.article_id.to_numpy(),
                                           tw.device)
    (tt if tt is not None else pd.DataFrame(columns=["anchor", "target_slot", "article_id", "sim"])
     ).to_parquet(paths["tt"], index=False)
    (ttc if ttc is not None else pd.DataFrame(columns=["anchor", "target_slot", "article_id", "sim"])
     ).to_parquet(paths["ttc"], index=False)
    key_cand.to_parquet(paths["key_sims"], index=False)
    info["artifact_seconds"] = round(time.time() - t0, 1)
    info["n_candidate_keys"] = int(len(key_cand))
    info["n_keys"] = int(len(keys))
    paths["info"].write_text(json.dumps(info, indent=2, default=str))
    log(f"  towers {week.start}: best epoch {info['best_epoch']} holdout R@12 "
        f"{info['best_holdout_recall@12']:.4f}, {len(key_cand):,} candidate keys "
        f"({info['artifact_seconds']:.0f}s)")
    return info


def main() -> None:
    import sys
    build_artifacts(sys.argv[1], log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()

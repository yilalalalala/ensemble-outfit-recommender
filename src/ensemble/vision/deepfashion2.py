"""DeepFashion2 consumer-to-shop adapter for FashionCLIP.

Stages:
  manifest  parse item annotations into identity-safe records
  embed     cache crop and segmentation/no-background FashionCLIP vectors
  train     fit small residual consumer/shop adapters on frozen vectors
  evaluate  exact-match Recall@1/5/10 on the official validation split
  all       run every stage in order

The raw dataset is research-only and remains under ``data/raw`` (gitignored).
No dataset image or derived crop is written to the repository.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from ensemble.config import load_config
from ensemble.vision.clip import load as load_clip, pad_square, device as clip_device

CATEGORIES = {
    1: "short_sleeve_top", 2: "long_sleeve_top", 3: "short_sleeve_outwear",
    4: "long_sleeve_outwear", 5: "vest", 6: "sling", 7: "shorts",
    8: "trousers", 9: "skirt", 10: "short_sleeve_dress",
    11: "long_sleeve_dress", 12: "vest_dress", 13: "sling_dress",
}


def paths(cfg=None) -> tuple[Path, Path]:
    cfg = cfg or load_config()
    return cfg.path("raw") / "deepfashion2", cfg.path("processed") / "deepfashion2"


def _split_root(raw: Path, split: str) -> Path:
    candidates = [raw / split, raw / "DeepFashion2" / split]
    for p in candidates:
        if (p / "annos").exists() and (p / "image").exists():
            return p
    raise FileNotFoundError(f"DeepFashion2 {split} not extracted under {raw}")


def build_manifest(splits=("train", "validation")) -> dict:
    """Parse every style>0 item; identity is split/pair_id/style/category."""
    raw, out = paths()
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in splits:
        root = _split_root(raw, split)
        rows = []
        for ap in sorted((root / "annos").glob("*.json")):
            d = json.loads(ap.read_text())
            source, pair_id = d.get("source"), int(d.get("pair_id", -1))
            if source not in ("user", "shop"):
                continue
            for key, item in d.items():
                if not key.startswith("item") or not isinstance(item, dict):
                    continue
                style, cat = int(item.get("style", 0)), int(item.get("category_id", 0))
                box = item.get("bounding_box")
                if style <= 0 or cat not in CATEGORIES or not box or len(box) != 4:
                    continue
                ident = f"{split}:{pair_id}:{style}:{cat}"
                rows.append({
                    "split": split, "image_id": ap.stem, "source": source,
                    "pair_id": pair_id, "style": style, "category_id": cat,
                    "category": CATEGORIES[cat], "identity": ident,
                    "x1": int(box[0]), "y1": int(box[1]), "x2": int(box[2]), "y2": int(box[3]),
                    "segmentation": json.dumps(item.get("segmentation", []), separators=(",", ":")),
                    "scale": int(item.get("scale", 0)), "occlusion": int(item.get("occlusion", 0)),
                    "zoom_in": int(item.get("zoom_in", 0)), "viewpoint": int(item.get("viewpoint", 0)),
                    "image_path": str(root / "image" / f"{ap.stem}.jpg"),
                })
        df = pd.DataFrame(rows)
        # An identity is trainable/evaluable only if both domains exist.
        valid = df.groupby("identity").source.nunique()
        df = df[df.identity.isin(valid[valid == 2].index)].reset_index(drop=True)
        df.insert(0, "row", np.arange(len(df), dtype=np.int32))
        df.to_parquet(out / f"{split}_manifest.parquet", index=False)
        counts[split] = {
            "records": len(df), "identities": int(df.identity.nunique()),
            "user_items": int((df.source == "user").sum()), "shop_items": int((df.source == "shop").sum()),
        }
    (out / "manifest_report.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=2))
    return counts


def _mask_crop(im: Image.Image, row, masked: bool) -> Image.Image:
    im = im.convert("RGB")
    x1, y1 = max(0, row.x1), max(0, row.y1)
    x2, y2 = min(im.width, row.x2), min(im.height, row.y2)
    if x2 <= x1 or y2 <= y1:
        return Image.new("RGB", (224, 224), "white")
    crop = im.crop((x1, y1, x2, y2))
    if not masked:
        return crop
    try:
        polygons = json.loads(row.segmentation)
    except (TypeError, json.JSONDecodeError):
        polygons = []
    mask = Image.new("L", crop.size, 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        if len(poly) >= 6:
            pts = [(float(poly[i]) - x1, float(poly[i + 1]) - y1) for i in range(0, len(poly) - 1, 2)]
            draw.polygon(pts, fill=255)
    if not mask.getbbox():
        return crop
    bg = Image.new("RGB", crop.size, "white")
    bg.paste(crop, mask=mask)
    return bg


class ItemCrops(Dataset):
    def __init__(self, frame: pd.DataFrame, processor, masked: bool):
        self.frame, self.processor, self.masked = frame, processor, masked

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        r = self.frame.iloc[i]
        with Image.open(r.image_path) as im:
            crop = pad_square(_mask_crop(im, r, self.masked))
            crop.thumbnail((336, 336))
            px = self.processor(images=crop, return_tensors="pt")["pixel_values"][0]
        return i, px


@torch.no_grad()
def embed_split(split: str) -> dict:
    _, out = paths()
    df = pd.read_parquet(out / f"{split}_manifest.parquet")
    model, proc = load_clip()
    dev = clip_device()
    stats = {}
    for masked, name in ((False, "crop"), (True, "nobg")):
        target = out / f"{split}_{name}_emb.npy"
        dim = int(model.config.projection_dim)
        arr = np.lib.format.open_memmap(target, mode="w+", dtype=np.float16, shape=(len(df), dim))
        dl = DataLoader(ItemCrops(df, proc, masked), batch_size=192, num_workers=4,
                        shuffle=False, persistent_workers=True)
        t0 = time.time()
        for step, (idx, px) in enumerate(dl):
            z = model.get_image_features(pixel_values=px.to(dev))
            z = getattr(z, "pooler_output", z)
            z = F.normalize(z, dim=-1).float().cpu().numpy().astype(np.float16)
            arr[idx.numpy()] = z
            if step % 100 == 0:
                print(f"  {split}/{name}: {min((step + 1) * 192, len(df)):,}/{len(df):,}", flush=True)
        arr.flush()
        stats[name] = {"records": len(df), "seconds": round(time.time() - t0, 1)}
    return stats


class ResidualAdapter(nn.Module):
    def __init__(self, dim: int = 512, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return F.normalize(x + self.net(x), dim=-1)


class DomainAdapters(nn.Module):
    def __init__(self, dim: int = 512, hidden: int = 256):
        super().__init__()
        self.user, self.shop = ResidualAdapter(dim, hidden), ResidualAdapter(dim, hidden)


def _identity_groups(df: pd.DataFrame):
    groups = []
    for (cat, ident), g in df.groupby(["category_id", "identity"], sort=False):
        u, s = g.index[g.source == "user"].to_numpy(), g.index[g.source == "shop"].to_numpy()
        if len(u) and len(s):
            groups.append((int(cat), u, s))
    return groups


def train_adapter(view: str = "crop") -> dict:
    cfg = load_config()
    _, out = paths(cfg)
    tc = cfg.visual_adapter
    df = pd.read_parquet(out / "train_manifest.parquet")
    emb = np.load(out / f"train_{view}_emb.npy", mmap_mode="r")
    groups = _identity_groups(df)
    by_cat = defaultdict(list)
    for g in groups:
        by_cat[g[0]].append(g)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = DomainAdapters(emb.shape[1], int(tc.hidden)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tc.lr), weight_decay=float(tc.weight_decay))
    rng = np.random.default_rng(int(tc.seed))
    history = []
    for epoch in range(int(tc.epochs)):
        losses = []
        for cat in rng.permutation(list(by_cat)):
            gs = by_cat[cat]
            order = rng.permutation(len(gs))
            for start in range(0, len(order), int(tc.batch_size)):
                batch = [gs[i] for i in order[start:start + int(tc.batch_size)]]
                if len(batch) < 2:
                    continue
                ui = np.asarray([rng.choice(g[1]) for g in batch])
                si = np.asarray([rng.choice(g[2]) for g in batch])
                u = torch.as_tensor(np.asarray(emb[ui], dtype=np.float32), device=dev)
                s = torch.as_tensor(np.asarray(emb[si], dtype=np.float32), device=dev)
                logits = model.user(u) @ model.shop(s).T / float(tc.temperature)
                labels = torch.arange(len(batch), device=dev)
                loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(float(loss.detach().cpu()))
        mean = float(np.mean(losses))
        history.append(mean)
        print(f"  {view} epoch {epoch + 1}: loss={mean:.5f}", flush=True)
    target = out / f"adapter_{view}.pt"
    torch.save({"state_dict": model.state_dict(), "dim": emb.shape[1], "hidden": int(tc.hidden),
                "view": view, "history": history}, target)
    return {"view": view, "identities": len(groups), "history": history, "artifact": str(target)}


@torch.no_grad()
def _adapt(model, x: np.ndarray, domain: str, batch=8192) -> np.ndarray:
    dev = next(model.parameters()).device
    fn = getattr(model, domain)
    return np.concatenate([fn(torch.as_tensor(x[i:i + batch].astype(np.float32), device=dev)).cpu().numpy()
                           for i in range(0, len(x), batch)])


def _metrics(query, gallery, qid, gid, qcat, gcat, k=10) -> dict:
    ranks, by_cat = [], defaultdict(list)
    for cat in sorted(set(qcat)):
        qi, gi = np.flatnonzero(qcat == cat), np.flatnonzero(gcat == cat)
        if not len(qi) or not len(gi):
            continue
        G = gallery[gi].astype(np.float32)
        for st in range(0, len(qi), 512):
            ii = qi[st:st + 512]
            score = query[ii].astype(np.float32) @ G.T
            top = np.argpartition(-score, min(k, len(gi)) - 1, axis=1)[:, :min(k, len(gi))]
            top_score = np.take_along_axis(score, top, axis=1)
            top = np.take_along_axis(top, np.argsort(-top_score, axis=1), axis=1)
            for row, q in zip(top, ii):
                hit = np.flatnonzero(gid[gi[row]] == qid[q])
                rank = int(hit[0] + 1) if len(hit) else k + 1
                ranks.append(rank); by_cat[int(cat)].append(rank)
    def pack(rr):
        a = np.asarray(rr)
        return {f"recall@{x}": float(np.mean(a <= x)) for x in (1, 5, 10)} | {
            "mrr@10": float(np.mean(np.where(a <= 10, 1 / a, 0))), "queries": len(a)}
    return {"overall": pack(ranks), "by_category": {CATEGORIES[c]: pack(v) for c, v in by_cat.items()}}


def evaluate() -> dict:
    _, out = paths()
    df = pd.read_parquet(out / "validation_manifest.parquet")
    user = df.source.eq("user").to_numpy(); shop = df.source.eq("shop").to_numpy()
    qid = df.identity.values[user]; gid = df.identity.values[shop]
    qcat = df.category_id.to_numpy()[user]; gcat = df.category_id.to_numpy()[shop]
    results = {}
    adapted = {}
    for view in ("crop", "nobg"):
        emb = np.load(out / f"validation_{view}_emb.npy", mmap_mode="r")
        results[f"fashionclip_{view}"] = _metrics(emb[user], emb[shop], qid, gid, qcat, gcat)
        ckpt = torch.load(out / f"adapter_{view}.pt", map_location="cpu")
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        mdl = DomainAdapters(ckpt["dim"], ckpt["hidden"]).to(dev)
        mdl.load_state_dict(ckpt["state_dict"]); mdl.eval()
        uq, sg = _adapt(mdl, emb[user], "user"), _adapt(mdl, emb[shop], "shop")
        adapted[view] = (uq, sg)
        results[f"adapter_{view}"] = _metrics(uq, sg, qid, gid, qcat, gcat)
    uq = F.normalize(torch.from_numpy(adapted["crop"][0] + adapted["nobg"][0]), dim=1).numpy()
    sg = F.normalize(torch.from_numpy(adapted["crop"][1] + adapted["nobg"][1]), dim=1).numpy()
    results["adapter_ensemble"] = _metrics(uq, sg, qid, gid, qcat, gcat)
    report = {"dataset": "DeepFashion2 validation", "protocol": "identity exact match; category-filtered gallery",
              "results": results}
    (out / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v["overall"] for k, v in results.items()}, indent=2))
    return report


def run_all():
    t0 = time.time()
    manifest = build_manifest()
    embedding = {s: embed_split(s) for s in ("train", "validation")}
    training = {v: train_adapter(v) for v in ("crop", "nobg")}
    evaluation = evaluate()
    report = {"manifest": manifest, "embedding": embedding, "training": training,
              "evaluation": evaluation, "seconds": round(time.time() - t0, 1)}
    _, out = paths()
    (out / "run.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "manifest": build_manifest((sys.argv[2],) if len(sys.argv) > 2 else ("train", "validation"))
    elif stage == "embed":
        embed_split(sys.argv[2] if len(sys.argv) > 2 else "train")
    elif stage == "train": train_adapter(sys.argv[2] if len(sys.argv) > 2 else "crop")
    elif stage == "evaluate": evaluate()
    elif stage == "all": run_all()
    else: raise SystemExit(f"unknown stage: {stage}")

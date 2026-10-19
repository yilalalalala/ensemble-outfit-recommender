"""How well are garments localised? The vision LLM's own boxes vs clothes-parsing boxes (vision/parsing.py), scored
against DeepFashion2 ground truth.

Each sampled validation photo (consumer "user" photos, the street-style case) is sent once to the detector; every
ground-truth garment is matched to the best-overlapping predicted item of a compatible category, and scored by IoU
(intersection over union) — with the LLM's box and with the parsed box for the same prediction. A garment with no
compatible prediction scores 0 under both (a detection miss, not a localisation error).

    python -m ensemble.vision.eval_boxes --n 40 --backend claude    # ~ $0.004 per photo (Haiku)
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from PIL import Image

from ensemble.config import load_config
from ensemble.vision import parsing
from ensemble.vision.deepfashion2 import paths
from ensemble.vision.outfit import _detect_llm

# DeepFashion2 category -> the detector categories that count as the same garment
COMPATIBLE = {
    "short sleeve top": {"top", "knitwear"}, "long sleeve top": {"top", "knitwear"}, "vest": {"top"}, "sling": {"top"},
    "short sleeve outwear": {"outerwear", "top"}, "long sleeve outwear": {"outerwear", "top", "knitwear"},
    "shorts": {"bottom"}, "trousers": {"bottom"}, "skirt": {"bottom"},
    "short sleeve dress": {"dress", "jumpsuit"}, "long sleeve dress": {"dress", "jumpsuit"},
    "vest dress": {"dress", "jumpsuit"}, "sling dress": {"dress", "jumpsuit"},
}


def iou(a, b) -> float:
    if not a or not b:
        return 0.0
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def sample(n: int, seed: int = 7) -> list[tuple[Path, list[dict]]]:
    root = paths()[0] / "validation"
    names = sorted(p.stem for p in (root / "annos").glob("*.json"))
    random.Random(seed).shuffle(names)
    out = []
    for name in names:
        a = json.loads((root / "annos" / f"{name}.json").read_text())
        items = [v for k, v in a.items() if k.startswith("item")]
        if a.get("source") == "user" and 1 <= len(items) <= 3 and all(it.get("occlusion", 1) <= 2 for it in items):
            out.append((root / "image" / f"{name}.jpg", items))
        if len(out) == n:
            break
    return out


def evaluate(n: int = 40, backend: str = "claude", log=print) -> dict:
    from ensemble.llm.client import get_client
    llm = get_client("vision", backend=backend)
    rows, t0 = [], time.time()
    for path, gt in sample(n):
        img = Image.open(path).convert("RGB")
        items, small = _detect_llm(img, llm, "eval_boxes")
        labels = parsing.segment(small)
        sx, sy = small.width / img.width, small.height / img.height
        for it in items:
            parsed = parsing.box_for(labels, it["category"], it.get("box"))
            own = it.get("box")
            it["parsed"] = parsed or own
            # "fill": keep the LLM's box, take the parse only where the LLM gave none (what auto does for Qwen-VL)
            it["filled"] = own or parsed
        for g in gt:
            gb = [g["bounding_box"][0] * sx, g["bounding_box"][1] * sy, g["bounding_box"][2] * sx, g["bounding_box"][3] * sy]
            cands = [it for it in items if it["category"] in COMPATIBLE[g["category_name"]]]
            best = max(cands, key=lambda it: max(iou(gb, it.get("box")), iou(gb, it["parsed"]), iou(gb, it["filled"])), default=None)
            rows.append({"image": path.name, "category": g["category_name"], "detected": best is not None,
                         "iou_llm": round(iou(gb, best.get("box")) if best else 0.0, 4),
                         "iou_parsed": round(iou(gb, best["parsed"]) if best else 0.0, 4),
                         "iou_filled": round(iou(gb, best["filled"]) if best else 0.0, 4)})
        log(f"  {path.name}: {len(items)} items · {time.time() - t0:.0f}s")
    det = [r for r in rows if r["detected"]]
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    summary = {
        "backend": backend, "images": n, "garments": len(rows), "detected": len(det),
        "mean_iou_llm": mean([r["iou_llm"] for r in det]), "mean_iou_parsed": mean([r["iou_parsed"] for r in det]),
        "mean_iou_filled": mean([r["iou_filled"] for r in det]),
        "hit_rate@0.5_llm": mean([r["iou_llm"] >= 0.5 for r in det]), "hit_rate@0.5_parsed": mean([r["iou_parsed"] >= 0.5 for r in det]),
        "hit_rate@0.5_filled": mean([r["iou_filled"] >= 0.5 for r in det]),
        "parser": parsing.MODEL, "seconds": round(time.time() - t0),
    }
    return {"summary": summary, "rows": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--backend", default="claude")
    args = ap.parse_args()
    res = evaluate(args.n, args.backend)
    out = load_config().path("reports") / "m7a" / f"box_localisation_{args.backend}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res["summary"], indent=2))
    print("wrote", out)

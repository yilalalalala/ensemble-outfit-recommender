"""Calibrate the visual-search relevance judge against human gold labels (D-019).

Candidate judges are scored on the owner's labels: agreement and Cohen's kappa, with a
garment-level (cluster) bootstrap 95% interval, because the 5 judgements of one garment are not
independent. Labels are split: round 1 is used to choose the judge (calibration set), round 2 only
to report the chosen judge (held-out set), the same discipline as validation/test weeks.

  python -m ensemble.vision.judge_calibration [round]
"""
from __future__ import annotations

import base64
import io
import json
import re
import sys
import time

import numpy as np
from PIL import Image, ImageOps

from ensemble.config import load_config
from ensemble.llm.client import BUDGET_USD, PRICES, _record, api_key, spend
from ensemble.vision.clip import image_path
from ensemble.vision.outfit import _resize_bytes

cfg = load_config()
OUT = cfg.path("reports") / "m7a"

GUIDELINE = """You label search results for a fashion visual-search system.
Image 1 is a {category} from a street photo. Images 2–6 are catalogue products.
For EACH product decide whether a shopper who wants the item in image 1 would accept that product as a
substitute. Mark 1 only if ALL three hold:
1. Same type of item (ankle boots ≠ sneakers; wide-leg trousers ≠ skinny; hoop earrings ≠ drop earrings).
2. Similar main colour (black ≈ dark grey; wine red ≈ red; wine red ≠ black; white ≠ black).
3. Similar style: fit, length and key design features.
Ignore brand, price, quality and fine details (buttons, stitching). Ignore how the product is photographed
(flat lay, model, close-up). Judge each product on its own; do not compare products with each other.
If unsure, mark 0. If image 1 is unreadable, mark all 0.
Return JSON only: {{"relevant": [a, b, c, d, e]}}"""

SHORT = """Image 1 is a {category} worn or carried in a street photo.
Images 2–6 are catalogue products. For each product, answer 1 if it is the same type of item with a similar colour
and style as the {category} in image 1 (a reasonable substitute a shopper would accept), else 0.
Judge only from what you see in image 1.
Return JSON only: {{"relevant": [a, b, c, d, e]}}"""

JUDGES = {
    "haiku_short": ("claude-haiku-4-5", SHORT, 384, 256),
    "haiku_guideline": ("claude-haiku-4-5", GUIDELINE, 512, 384),
    "sonnet_guideline": ("claude-sonnet-5", GUIDELINE, 512, 384),
    "opus_guideline": ("claude-opus-5", GUIDELINE, 512, 384),
}


def _b64(img: Image.Image, side: int) -> str:
    img = img.copy().convert("RGB")
    img.thumbnail((side, side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def tasks(round_: int) -> tuple[list[dict], dict]:
    name = "human_labels.json" if round_ == 1 else f"human_labels_round{round_}.json"
    labels = json.loads((OUT / name).read_text())
    rows = {(r["file"], r["garment"], r["mode"]): r for r in json.loads((OUT / "matches_judged.json").read_text())}
    out = []
    for tid, lab in labels.items():
        f, g, m = tid.split("|")
        r = rows[(f, int(g), m)]
        out.append({"task_id": tid, "file": f, "garment": int(g), "category": r["category"],
                    "candidates": r["candidates"], "human": lab, "stored_judge": r["relevant"]})
    return out, labels


def judge_one(client, model: str, prompt: str, qside: int, cside: int, t: dict) -> tuple[list[int] | None, float]:
    det = json.loads((OUT / "detections_ollama.json").read_text())
    _, small = _resize_bytes(Image.open(cfg.path("raw") / "outfit_photos" / t["file"]))
    crop = small.crop(det[t["file"]]["items"][t["garment"]]["box"])
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(crop, qside)}}]
    for a in t["candidates"]:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": _b64(Image.open(image_path(cfg, a)), cside)}})
    content.append({"type": "text", "text": prompt.format(category=t["category"])})
    kw = {"output_config": {"effort": "low"}} if model.startswith(("claude-opus", "claude-sonnet-5")) else {}
    r = client.messages.create(model=model, max_tokens=2000, messages=[{"role": "user", "content": content}], **kw)
    pin, pout = PRICES[model]
    cost = (r.usage.input_tokens * pin + r.usage.output_tokens * pout) / 1e6
    text = "".join(b.text for b in r.content if b.type == "text")
    m = re.search(r"\[[^\]]*\]", text)
    try:
        return ([int(x) for x in json.loads(m.group(0))][:5] if m else None), cost
    except (json.JSONDecodeError, ValueError):
        return None, cost


def agreement(ts: list[dict], key: str, n_boot: int = 2000, seed: int = 0) -> dict:
    def stats(sub):
        pairs = [(h, j) for t in sub if t.get(key) for h, j in zip(t["human"], t[key])]
        if not pairs:
            return float("nan"), float("nan")
        a = np.asarray(pairs)
        po = float(np.mean(a[:, 0] == a[:, 1]))
        ph, pj = a[:, 0].mean(), a[:, 1].mean()
        pe = ph * pj + (1 - ph) * (1 - pj)
        return po, float((po - pe) / (1 - pe)) if pe < 1 else float("nan")
    po, kappa = stats(ts)
    garments = sorted({(t["file"], t["garment"]) for t in ts})
    by_g = {g: [t for t in ts if (t["file"], t["garment"]) == g] for g in garments}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):   # cluster bootstrap: resample garments, not single judgements
        sample = [t for g in rng.choice(len(garments), len(garments)) for t in by_g[garments[g]]]
        boots.append(stats(sample))
    b = np.asarray(boots)
    return {"agreement": po, "agreement_ci95": [float(np.nanpercentile(b[:, 0], 2.5)), float(np.nanpercentile(b[:, 0], 97.5))],
            "kappa": kappa, "kappa_ci95": [float(np.nanpercentile(b[:, 1], 2.5)), float(np.nanpercentile(b[:, 1], 97.5))],
            "judgements": sum(len(t["human"]) for t in ts if t.get(key))}


def run(round_: int = 1, judges: list[str] | None = None) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key())
    ts, _ = tasks(round_)
    results = {"stored_haiku_batch": agreement(ts, "stored_judge")}
    for name in judges or list(JUDGES):
        model, prompt, qs, cs = JUDGES[name]
        if spend()["total_usd"] + 0.8 > BUDGET_USD:
            print("budget guard: stopping before", name)
            break
        total = 0.0
        for t in ts:
            t[name], c = judge_one(client, model, prompt, qs, cs, t)
            total += c
        _record(total, "judge_calibration")
        results[name] = agreement(ts, name) | {"cost_usd": round(total, 4)}
        r = results[name]
        print(f"{name:18s} agreement {r['agreement']:.3f} {np.round(r['agreement_ci95'], 3)}  "
              f"kappa {r['kappa']:.3f} {np.round(r['kappa_ci95'], 3)}  ${total:.3f}", flush=True)
    out = OUT / f"judge_calibration_round{round_}.json"
    out.write_text(json.dumps({"results": results, "tasks": ts}, indent=1, default=str))
    return results


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 1, sys.argv[2].split(",") if len(sys.argv) > 2 else None)

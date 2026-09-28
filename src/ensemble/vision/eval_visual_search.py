"""M7a evaluation: garment detection and street-to-shop matching on the outfit photo set.

  python -m ensemble.vision.eval_visual_search detect ollama     VLM detections (cached)
  python -m ensemble.vision.eval_visual_search detect claude 30  paid comparison on a subset
  python -m ensemble.vision.eval_visual_search match             top-5 per garment for each query mode
  python -m ensemble.vision.eval_visual_search judge             relevance by Claude Haiku (Batch API, 50% off)
  python -m ensemble.vision.eval_visual_search report            Precision@5 overall and by segment

Relevance = same garment type, similar colour and style: a reasonable substitute. Street photos are not
H&M products, so exact-match accuracy is undefined; Precision@5 of judged relevance is the metric.
"""
from __future__ import annotations

import base64
import csv
import io
import json
import re
import sys
import time

import numpy as np
from PIL import Image, ImageOps

from ensemble.config import load_config
from ensemble.llm.client import ClaudeClient, OllamaClient, _record, api_key
from ensemble.vision.outfit import CATEGORIES, detect, match

MODES = ["photo", "crop", "text", "crop+text", "text+0.3crop", "crop_nobg", "crop_adapter", "crop_nobg_adapter", "text_rerank_image"]
cfg = load_config()
PHOTOS = cfg.path("raw") / "outfit_photos"
OUT = cfg.path("reports") / "m7a"


def manifest() -> dict[str, dict]:
    return {r["file"]: r for r in csv.DictReader(open(PHOTOS / "MANIFEST.csv"))}


def stage_detect(backend: str, limit: int | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    llm = ClaudeClient("claude-opus-5") if backend == "claude" else OllamaClient("qwen2.5vl:7b")
    files = sorted(manifest())
    if limit:  # a fixed, stratified subset: every other photo
        files = files[::max(1, len(files) // limit)][:limit]
    path = OUT / f"detections_{backend}.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for f in files:
        if f in res:
            continue
        t = time.time()
        try:
            items, small = detect(Image.open(PHOTOS / f), llm, purpose="detect")
            res[f] = {"items": items, "size": list(small.size), "seconds": round(time.time() - t, 1)}
        except Exception as e:
            res[f] = {"items": [], "error": str(e), "seconds": round(time.time() - t, 1)}
        print(f"  {backend} {f}: {[i['category'] for i in res[f]['items']]} ({res[f]['seconds']}s)", flush=True)
        path.write_text(json.dumps(res, indent=1))


def stage_match(modes: list[str] | None = None) -> None:
    """Compute top-5 for ``modes`` (default all), keeping rows already computed for other modes."""
    det = json.loads((OUT / "detections_ollama.json").read_text())
    modes = modes or MODES
    path = OUT / "matches.json"
    rows = [r for r in json.loads(path.read_text()) if r["mode"] not in modes] if path.exists() else []
    for f, d in det.items():
        img = ImageOps.exif_transpose(Image.open(PHOTOS / f)).convert("RGB")
        img.thumbnail((1024, 1024))
        for gi, g in enumerate(d["items"]):
            for mode in modes:
                m = match(g, img, mode, 5)
                rows.append({"file": f, "garment": gi, "category": g["category"], "colour": g["colour"],
                             "description": g["description"], "box": g["box"], "mode": mode,
                             "candidates": m.article_id.tolist()})
    (OUT / "matches.json").write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} (garment, mode) queries")


def _b64(img: Image.Image, side: int) -> str:
    img = img.copy()
    img.thumbnail((side, side))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# The judge sees only pixels and the coarse category, never the VLM's text description: the "text" query
# mode searches with that description, so showing it to the judge would favour that mode (evaluation leak).
JUDGE = """Image 1 is a {category} worn or carried in a street photo{crop_note}.
Images 2–6 are catalogue products. For each product, answer 1 if it is the same type of item with a similar colour
and style as the {category} in image 1 (a reasonable substitute a shopper would accept), else 0.
Judge only from what you see in image 1.
Return JSON only: {{"relevant": [a, b, c, d, e]}}"""


def stage_judge() -> None:
    """Judge rows without a label yet (labels already in matches_judged.json are kept)."""
    import anthropic
    from ensemble.vision.clip import image_path
    rows = json.loads((OUT / "matches.json").read_text())
    done_path = OUT / "matches_judged.json"
    if done_path.exists():
        prev = {(r["file"], r["garment"], r["mode"]): r.get("relevant") for r in json.loads(done_path.read_text())}
        for r in rows:
            r["relevant"] = prev.get((r["file"], r["garment"], r["mode"]))
    todo = [i for i, r in enumerate(rows) if r.get("relevant") is None]
    client = anthropic.Anthropic(api_key=api_key())
    reqs = []
    for i in todo:
        r = rows[i]
        img = ImageOps.exif_transpose(Image.open(PHOTOS / r["file"])).convert("RGB")
        img.thumbnail((1024, 1024))
        q = img.crop(r["box"]) if r["box"] else img
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(q, 384)}}]
        for a in r["candidates"]:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                        "data": _b64(Image.open(image_path(cfg, a)), 256)}})
        content.append({"type": "text", "text": JUDGE.format(
            crop_note="" if r["box"] else " (whole photo; focus on that item)", category=r["category"])})
        reqs.append({"custom_id": f"q{i}", "params": {"model": "claude-haiku-4-5", "max_tokens": 100,
                                                       "messages": [{"role": "user", "content": content}]}})
    from ensemble.llm.client import BUDGET_USD, spend
    if spend()["total_usd"] + 1.0 > BUDGET_USD:
        raise RuntimeError("judging could exceed the LLM budget")
    labels, cost = {}, 0.0
    for c in range(0, len(reqs), 400):   # keep each batch request body well under the size limit
        batch = client.messages.batches.create(requests=reqs[c:c + 400])
        print("batch", batch.id, len(reqs[c:c + 400]), "requests", flush=True)
        while client.messages.batches.retrieve(batch.id).processing_status != "ended":
            time.sleep(20)
        for res in client.messages.batches.results(batch.id):
            if res.result.type != "succeeded":
                continue
            msg = res.result.message
            cost += (msg.usage.input_tokens * 1.0 + msg.usage.output_tokens * 5.0) / 1e6 * 0.5
            text = "".join(b.text for b in msg.content if b.type == "text")
            m = re.search(r"\[[^\]]*\]", text)
            try:
                labels[res.custom_id] = [int(x) for x in json.loads(m.group(0))][:5] if m else None
            except (json.JSONDecodeError, ValueError):
                labels[res.custom_id] = None
    _record(cost, "visual_judge")
    for i in todo:
        rows[i]["relevant"] = labels.get(f"q{i}")
    (OUT / "matches_judged.json").write_text(json.dumps(rows, indent=1))
    # keep the first (biased) run for the record

    print(f"judged {sum(r['relevant'] is not None for r in rows)}/{len(rows)}; cost ${cost:.3f}")


def stage_report() -> dict:
    rows = [r for r in json.loads((OUT / "matches_judged.json").read_text()) if r.get("relevant")]
    man = manifest()

    def p5(sub):
        return float(np.mean([sum(r["relevant"][:5]) / 5 for r in sub])) if sub else float("nan")

    out = {"n_queries_per_mode": {m: sum(r["mode"] == m for r in rows) for m in MODES}}
    out["precision@5"] = {m: p5([r for r in rows if r["mode"] == m]) for m in MODES}
    out["precision@1"] = {m: float(np.mean([r["relevant"][0] for r in rows if r["mode"] == m])) for m in MODES}
    seg = {}
    for key, fn in {"source": lambda r: man[r["file"]]["source"], "photo_type": lambda r: man[r["file"]]["type"],
                    "text_in_image": lambda r: "yes" if man[r["file"]]["text_in_image"] else "no",
                    "category": lambda r: r["category"], "has_box": lambda r: "box" if r["box"] else "no box"}.items():
        vals = sorted({fn(r) for r in rows})
        seg[key] = {v: {m: p5([r for r in rows if r["mode"] == m and fn(r) == v]) for m in MODES} | {
            "n": sum(1 for r in rows if r["mode"] == "crop+text" and fn(r) == v)} for v in vals}
    out["by_segment"] = seg
    det = json.loads((OUT / "detections_ollama.json").read_text())
    out["detection"] = {"photos": len(det), "garments": sum(len(d["items"]) for d in det.values()),
                        "valid_box_share": float(np.mean([bool(i["box"]) for d in det.values() for i in d["items"]])),
                        "median_seconds": float(np.median([d["seconds"] for d in det.values()]))}
    if (OUT / "detections_claude.json").exists():
        dc = json.loads((OUT / "detections_claude.json").read_text())
        common = [f for f in dc if f in det]
        agree = []
        for f in common:
            a = {CATEGORIES[i["category"]][0] for i in det[f]["items"]}
            b = {CATEGORIES[i["category"]][0] for i in dc[f]["items"]}
            agree.append(len(a & b) / len(a | b) if a | b else 1.0)
        out["detection_vs_claude"] = {"photos": len(common), "slot_jaccard_mean": float(np.mean(agree)) if agree else None,
                                      "claude_garments": sum(len(dc[f]["items"]) for f in common),
                                      "qwen_garments": sum(len(det[f]["items"]) for f in common),
                                      "claude_valid_box_share": float(np.mean([bool(i["box"]) for f in common for i in dc[f]["items"]] or [0]))}
    hs = OUT / "reviewer_spot_check_claude.json"  # labelled by Claude, not a human (see D-019)
    if hs.exists():
        judged = {(r["file"], r["garment"], r["mode"]): r["relevant"] for r in rows}
        pairs = [(h, judged.get((h["file"], h["garment"], h["mode"]))) for h in json.loads(hs.read_text())]
        pairs = [(h["label"], j) for h, j in pairs if j]
        flat = [(a, b) for hh, jj in pairs for a, b in zip(hh, jj)]
        out["judge_vs_reviewer"] = {"reviewer": "Claude (not a human)", "items": len(flat),
                                    "agreement": float(np.mean([a == b for a, b in flat])),
                                    "reviewer_p5": {m: float(np.mean([sum(h["label"]) / 5 for h in json.loads(hs.read_text()) if h["mode"] == m]))
                                                    for m in ("crop", "text")}}
    hl = OUT / "human_labels.json"
    if hl.exists():
        judged = {(r["file"], r["garment"], r["mode"]): r["relevant"] for r in rows}
        flat, human_p5 = [], {}
        for tid, lab in json.loads(hl.read_text()).items():
            f, g, m = tid.split("|")
            j = judged.get((f, int(g), m))
            if j:
                flat += list(zip(lab, j))
            human_p5.setdefault(m, []).append(sum(lab) / 5)
        out["judge_vs_human"] = {"labeller": "project owner", "items": len(flat),
                                 "agreement": float(np.mean([a == b for a, b in flat])) if flat else None,
                                 "human_p5": {m: float(np.mean(v)) for m, v in human_p5.items()}}
    (cfg.path("reports") / "m7a_visual_search.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "detect":
        stage_detect(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
    else:
        if stage == "match":
            stage_match(sys.argv[2].split(",") if len(sys.argv) > 2 else None)
        else:
            {"judge": stage_judge, "report": stage_report}[stage]()

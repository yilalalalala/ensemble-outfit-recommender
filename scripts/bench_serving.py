"""Local performance benchmark of the Track B serving API (reports/track_b_production/bench_*.json).

  python scripts/bench_serving.py [--bundle data/processed/serving_bundles] [--out NAME] [--requests 2000]

Everything is measured on this machine against a real uvicorn process (one worker):

- startup: process start -> /readyz 200 (bundle load + validation);
- RSS of the server process, sampled every 100 ms (steady state and peak);
- Complete the Look latency p50/p95/p99 and throughput at concurrency 1, 4 and 16, for a
  reproducible workload (seeded): anchors drawn with a Zipf(1.1) skew over the bundle's keys,
  customers 50% with a profile, 25% unknown ids, 25% anonymous;
  * cold  = the cache disabled (every request computes), i.e. compute cost;
  * warm  = cache enabled after one warm-up pass of the same workload (hit rate reported);
  * first = the very first request after startup;
- visual search (FashionCLIP): first call (model load) and warm latency, concurrency 1 and 4;
- exact vector search over the visual index (in-process, NumPy) as the ANN decision baseline;
- the fallback-level / provenance / personalization mix reported by /api/v2/metrics;
- error and timeout counts; bundle file sizes.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from serve_smoke import free_port, start, wait_ready  # noqa: E402


def rss_mb(pid: int) -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) / 1024 if out else float("nan")


class RssSampler(threading.Thread):
    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.samples, self.stop = pid, [], False

    def run(self):
        while not self.stop:
            self.samples.append(rss_mb(self.pid))
            time.sleep(0.1)


def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else float("nan")


def summarize(lat_ms, wall_s, errors, timeouts):
    return {"n": len(lat_ms), "p50_ms": pct(lat_ms, 50), "p95_ms": pct(lat_ms, 95), "p99_ms": pct(lat_ms, 99),
            "mean_ms": float(statistics.mean(lat_ms)) if lat_ms else float("nan"),
            "throughput_rps": len(lat_ms) / wall_s if wall_s else float("nan"), "errors": errors,
            "timeouts": timeouts, "error_rate": (errors + timeouts) / max(1, len(lat_ms) + errors + timeouts)}


async def run_load(base: str, reqs: list[dict], concurrency: int, path="/api/v2/complete-the-look", timeout=10.0):
    lat, errors, timeouts = [], 0, 0
    q = asyncio.Queue()
    for r in reqs:
        q.put_nowait(r)
    async with httpx.AsyncClient(base_url=base, timeout=timeout,
                                 limits=httpx.Limits(max_connections=concurrency)) as client:
        async def worker():
            nonlocal errors, timeouts
            while True:
                try:
                    r = q.get_nowait()
                except asyncio.QueueEmpty:
                    return
                t = time.perf_counter()
                try:
                    resp = await client.get(path, params=r)
                    if resp.status_code == 200:
                        lat.append((time.perf_counter() - t) * 1000)
                    else:
                        errors += 1
                except httpx.TimeoutException:
                    timeouts += 1
        t0 = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(concurrency)])
        wall = time.perf_counter() - t0
    return summarize(lat, wall, errors, timeouts)


def workload(bundle_dir: Path, n: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    keys = np.load(bundle_dir / "keys.npz")
    anchors = np.unique(keys["anchor"])
    rank = rng.permutation(len(anchors))
    w = 1.0 / (np.arange(1, len(anchors) + 1) ** 1.1)
    w /= w.sum()
    pick = anchors[rank][rng.choice(len(anchors), size=n, p=w)]
    con = sqlite3.connect(f"file:{bundle_dir / 'profiles.sqlite'}?mode=ro", uri=True)
    custs = np.array([r[0] for r in con.execute(
        "SELECT customer_idx FROM profiles WHERE customer_idx % 211 = 7 LIMIT 20000")])
    if not len(custs):
        custs = np.array([r[0] for r in con.execute("SELECT customer_idx FROM profiles LIMIT 20000")])
    con.close()
    reqs = []
    for i, a in enumerate(pick):
        u = rng.random()
        r = {"anchor": int(a), "k": 8}
        if u < 0.5:
            r["customer"] = int(rng.choice(custs))
        elif u < 0.75:
            r["customer"] = int(2_000_000_000 + rng.integers(0, 10**6))     # unknown id: no profile
        reqs.append(r)
    return reqs


def visual_bench(base: str, bundle_dir: Path, n: int = 40) -> dict:
    from ensemble.config import load_config
    cfg = load_config()
    ids = np.load(bundle_dir / "visual" / "live_ids.npy")
    rng = np.random.default_rng(1)
    imgs = []
    for a in rng.choice(ids, size=min(n * 3, len(ids)), replace=False):
        s = f"{int(a):010d}"
        p = cfg.path("raw") / "images" / s[:3] / f"{s}.jpg"
        if p.exists():
            imgs.append(p.read_bytes())
        if len(imgs) == n:
            break
    if not imgs:
        return {"skipped": "no product images available"}
    out = {}
    with httpx.Client(base_url=base, timeout=120) as c:
        t = time.perf_counter()
        r = c.post("/api/v2/visual-search", files={"photo": ("a.jpg", imgs[0], "image/jpeg")}, data={"category": "top"})
        out["first_call_ms"] = (time.perf_counter() - t) * 1000
        out["first_call_status"] = r.status_code
        if r.status_code != 200:
            out["error"] = r.json()
            return out
        out["mode"] = r.json()["mode"]
        for conc in (1, 4):
            lat, errors = [], 0

            def one(b):
                nonlocal errors
                t = time.perf_counter()
                rr = c.post("/api/v2/visual-search", files={"photo": ("a.jpg", b, "image/jpeg")},
                            data={"category": "top", "k": "8"})
                if rr.status_code == 200:
                    lat.append((time.perf_counter() - t) * 1000)
                else:
                    errors += 1
            t0 = time.perf_counter()
            threads = []
            for i in range(0, len(imgs), conc):
                threads = [threading.Thread(target=one, args=(b,)) for b in imgs[i:i + conc]]
                for th in threads:
                    th.start()
                for th in threads:
                    th.join()
            out[f"warm_c{conc}"] = summarize(lat, time.perf_counter() - t0, errors, 0)
    return out


def exact_search_bench(bundle_dir: Path) -> dict:
    v = np.load(bundle_dir / "visual" / "clip.npy").astype(np.float32)
    q = v[np.random.default_rng(2).choice(len(v), 200)]
    t = time.perf_counter()
    for x in q:
        s = v @ x
        np.argpartition(-s, 8)[:8]
    ms = (time.perf_counter() - t) * 1000 / len(q)
    return {"n_vectors": int(len(v)), "dim": int(v.shape[1]), "exact_top8_ms_per_query": ms,
            "index_mb": v.nbytes / 1e6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=ROOT / "data/processed/serving_bundles")
    ap.add_argument("--out", default="bench_serving")
    ap.add_argument("--requests", type=int, default=2000)
    a = ap.parse_args()
    from ensemble.serving.runtime import resolve
    bdir = resolve(a.bundle)
    reqs = workload(bdir, a.requests)
    res = {"bundle": bdir.name, "machine": {"platform": sys.platform, "cpu": subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
        "ram_gb": int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30},
        "workload": {"requests": len(reqs), "zipf_s": 1.1, "share_known_customer": 0.5, "share_unknown_id": 0.25,
                     "share_anonymous": 0.25, "k": 8, "workers": 1},
        "bundle_files_mb": {str(p.relative_to(bdir)): round(p.stat().st_size / 1e6, 2)
                            for p in sorted(bdir.rglob("*")) if p.is_file()}}
    res["bundle_total_mb"] = round(sum(res["bundle_files_mb"].values()), 1)
    for mode, cache in (("cold", "0"), ("warm", "200000")):
        port = free_port()
        base = f"http://127.0.0.1:{port}"
        t0 = time.time()
        proc = start(a.bundle, port, sys.executable, {"ENSEMBLE_CACHE_SIZE": cache})
        sampler = RssSampler(proc.pid)
        sampler.start()
        try:
            ready = wait_ready(base, proc)
            rss_ready = rss_mb(proc.pid)
            t = time.perf_counter()
            first = httpx.get(f"{base}/api/v2/complete-the-look", params=reqs[0], timeout=30)
            first_ms = (time.perf_counter() - t) * 1000
            block = {"startup_to_ready_s": ready, "rss_after_ready_mb": rss_ready,
                     "first_request_ms": first_ms, "first_request_status": first.status_code}
            if mode == "warm":
                asyncio.run(run_load(base, reqs, 8))          # warm-up pass (fills the cache)
            for conc in (1, 4, 16):
                block[f"c{conc}"] = asyncio.run(run_load(base, reqs, conc))
            m = httpx.get(f"{base}/api/v2/metrics").json()
            block["cache"] = m.get("cache")
            block["mix"] = {c["name"] + ":" + ",".join(f"{k}={v}" for k, v in sorted(c["labels"].items())): c["value"]
                            for c in m["counters"] if c["name"] in ("ctl_modules", "ctl_items", "ctl_cache")}
            if mode == "warm":
                block["visual"] = visual_bench(base, bdir)
            block["rss_steady_mb"] = rss_mb(proc.pid)
        finally:
            sampler.stop = True
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        block["rss_peak_mb"] = float(np.nanmax(sampler.samples)) if sampler.samples else float("nan")
        block["wall_s"] = round(time.time() - t0, 1)
        res[mode] = block
        print(json.dumps({mode: {k: v for k, v in block.items() if k not in ("mix",)}}, indent=1, default=float),
              flush=True)
    res["exact_vector_search"] = exact_search_bench(bdir) if (bdir / "visual" / "clip.npy").exists() else None
    out = ROOT / "reports" / "track_b_production" / f"{a.out}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

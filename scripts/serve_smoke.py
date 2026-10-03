"""End-to-end smoke test of the serving API in a real uvicorn process.

  python scripts/serve_smoke.py                       # synthetic fixture bundle (no H&M data needed)
  python scripts/serve_smoke.py --bundle data/processed/serving_bundles   # the real, CURRENT bundle

Starts the server on a free port, waits for readiness, checks health, readiness, metadata,
anonymous and personalized Complete the Look, an error response and the metrics endpoint,
then stops the server. Exits non-zero on the first failed check.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start(bundle: Path, port: int, python: str, env_extra: dict | None = None,
          log_path: Path | None = None) -> subprocess.Popen:
    """Start the API. Its JSON request log goes to a file: an undrained pipe would fill up and
    block the server on its next log line."""
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "ENSEMBLE_BUNDLE": str(bundle), **(env_extra or {})}
    log_path = log_path or Path(tempfile.gettempdir()) / f"ensemble-serve-{port}.log"
    fh = open(log_path, "w")
    p = subprocess.Popen([python, "-m", "uvicorn", "ensemble.api.app:app", "--host", "127.0.0.1",
                          "--port", str(port), "--log-level", "warning"], cwd=ROOT, env=env,
                         stdout=subprocess.DEVNULL, stderr=fh)
    p.log_path = log_path
    return p


def wait_ready(base: str, proc: subprocess.Popen, timeout: float = 180.0) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"server exited early: {Path(proc.log_path).read_text()[-2000:]}")
        try:
            if httpx.get(f"{base}/readyz", timeout=2).status_code == 200:
                return time.time() - t0
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise SystemExit("server did not become ready in time")


def pick(bundle: Path) -> tuple[int, int]:
    """An anchor with pools and a customer with a profile, from the bundle itself."""
    import numpy as np
    from ensemble.serving.runtime import resolve
    b = resolve(bundle)
    keys = np.load(b / "keys.npz")
    anchor = int(keys["anchor"][len(keys["anchor"]) // 3])
    con = sqlite3.connect(f"file:{b / 'profiles.sqlite'}?mode=ro", uri=True)
    customer = con.execute("SELECT customer_idx FROM profiles ORDER BY customer_idx LIMIT 1").fetchone()[0]
    con.close()
    return anchor, int(customer)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    tmp = None
    if a.bundle is None:
        from ensemble.serving import fixture
        tmp = tempfile.TemporaryDirectory()
        a.bundle = fixture.build(Path(tmp.name) / "fixture-v1")
    anchor, customer = pick(a.bundle)
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    proc = start(a.bundle, port, a.python)
    checks = []

    def check(name, cond, detail=""):
        checks.append({"check": name, "ok": bool(cond), "detail": detail})
        if not cond:
            raise AssertionError(f"{name}: {detail}")

    try:
        ready_s = wait_ready(base, proc)
        check("healthz", httpx.get(f"{base}/healthz").json()["status"] == "ok")
        r = httpx.get(f"{base}/readyz").json()
        check("readyz", r["ready"], r.get("bundle_version"))
        m = httpx.get(f"{base}/api/v2/meta").json()
        check("meta versions", all(m.get(k) for k in ("bundle_version", "model_version", "catalog_version")))
        anon = httpx.get(f"{base}/api/v2/complete-the-look", params={"anchor": anchor, "k": 8}).json()
        check("anonymous CTL", anon["modules"] and not anon["personalized"],
              f"{len(anon['modules'])} modules, levels {sorted({x['fallback_level'] for x in anon['modules']})}")
        pers = httpx.get(f"{base}/api/v2/complete-the-look",
                         params={"anchor": anchor, "k": 8, "customer": customer}).json()
        check("personalized CTL", pers["personalized"] and pers["customer_known"])
        check("provenance on every row", all(it["evidence_types"] is not None and it["provenance"]
                                             for mm in pers["modules"] for it in mm["items"]))
        e = httpx.get(f"{base}/api/v2/complete-the-look", params={"anchor": 1})
        check("error schema", e.status_code == 404 and e.json()["error"]["code"] == "unknown_article"
              and e.json()["error"]["request_id"] == e.headers["X-Request-ID"])
        met = httpx.get(f"{base}/metrics").text
        check("metrics", "ensemble_requests" in met and "ensemble_ready 1" in met)
        print(json.dumps({"ok": True, "ready_seconds": round(ready_s, 2), "bundle": r["bundle_version"],
                          "checks": checks}, indent=1))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        if tmp:
            tmp.cleanup()


if __name__ == "__main__":
    main()

"""Process-level serving state: the active bundle, readiness, atomic reload, metrics.

Readiness is false until a bundle has loaded and passed validation (``runtime.Bundle``). A
reload builds the new ``Recommender`` completely, validates it, and only then swaps the
reference under a lock, so in-flight requests finish on the old bundle and no request ever
sees a half-loaded one. Every cache key contains the bundle and availability versions, and the
swap also starts a fresh cache, so a stale recommendation cannot survive a bundle change.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from ensemble.serving.observability import Metrics
from ensemble.serving.runtime import Bundle, BundleError, Recommender


def default_bundle_path() -> Path:
    env = os.environ.get("ENSEMBLE_BUNDLE")
    if env:
        return Path(env)
    from ensemble.config import load_config
    return load_config().path("processed") / "serving_bundles"


class ServingState:
    def __init__(self):
        self.lock = threading.Lock()
        self.rec: Recommender | None = None
        self.visual = None
        self.error: str | None = None
        self.loaded_at: float | None = None
        self.started = time.time()
        self.metrics = Metrics()
        self.load_history: list[dict] = []

    @property
    def ready(self) -> bool:
        return self.rec is not None

    def load(self, path: Path | None = None, verify_hashes: bool = False) -> dict:
        """Load and validate a bundle; swap it in only if everything passed."""
        from ensemble.serving.visual import VisualSearch
        path = Path(path) if path else default_bundle_path()
        t0 = time.time()
        try:
            bundle = Bundle(path, verify_hashes=verify_hashes)
            rec = Recommender(bundle, cache_size=int(os.environ.get("ENSEMBLE_CACHE_SIZE", "20000")),
                              personalization=os.environ.get("ENSEMBLE_PERSONALIZATION", "1") != "0")
            visual = VisualSearch(bundle, timeout_s=float(os.environ.get("ENSEMBLE_VISUAL_TIMEOUT_S", "20")))
        except BundleError as e:
            self.error = str(e)
            self.metrics.inc("bundle_load_failures")
            self.load_history.append({"path": str(path), "ok": False, "error": str(e), "at": time.time()})
            raise
        with self.lock:
            old = self.rec
            self.rec, self.visual, self.error, self.loaded_at = rec, visual, None, time.time()
        if old is not None:
            old.cache.clear()
        info = {"path": str(bundle.path), "ok": True, "bundle_version": bundle.version,
                "load_seconds": round(time.time() - t0, 3), "at": time.time()}
        self.load_history.append(info)
        self.metrics.inc("bundle_loads")
        return info

    def try_load(self, path: Path | None = None) -> None:
        try:
            self.load(path)
        except BundleError:
            pass                  # readiness stays false; /readyz reports self.error

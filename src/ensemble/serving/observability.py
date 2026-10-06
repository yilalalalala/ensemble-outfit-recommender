"""Local stand-ins for production observability: structured request logs and a metrics registry.

- **Structured logs**: one JSON line per request (request id, endpoint, status, latency, model /
  catalogue / bundle versions, result count, fallback levels, personalization flag, cache
  outcome, error class). Customer ids are logged only as a salted hash; image bytes, raw
  uploads and purchase histories are never logged.
- **Metrics**: counters and fixed-bucket latency histograms, exposed in the Prometheus text
  format (``/metrics``) and as JSON (``/api/v2/metrics``).
"""
from __future__ import annotations

import bisect
import hashlib
import json
import logging
import os
import threading
import time

BUCKETS_MS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
_SALT = os.environ.get("ENSEMBLE_LOG_SALT", "ensemble-local")


def pseudonymize(customer_idx) -> str | None:
    if customer_idx is None:
        return None
    return hashlib.sha256(f"{_SALT}:{customer_idx}".encode()).hexdigest()[:12]


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.counters: dict[tuple, float] = {}
        self.hist: dict[tuple, list] = {}

    def inc(self, name: str, value: float = 1.0, **labels) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self.lock:
            self.counters[key] = self.counters.get(key, 0.0) + value

    def observe_ms(self, name: str, ms: float, **labels) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self.lock:
            h = self.hist.setdefault(key, [[0] * (len(BUCKETS_MS) + 1), 0.0, 0])
            h[0][bisect.bisect_left(BUCKETS_MS, ms)] += 1
            h[1] += ms
            h[2] += 1

    def snapshot(self) -> dict:
        with self.lock:
            c = [{"name": n, "labels": dict(lb), "value": v} for (n, lb), v in sorted(self.counters.items())]
            h = [{"name": n, "labels": dict(lb), "count": x[2], "sum_ms": round(x[1], 3),
                  "buckets_ms": dict(zip([*map(str, BUCKETS_MS), "+Inf"], x[0]))}
                 for (n, lb), x in sorted(self.hist.items())]
        return {"counters": c, "histograms": h}

    def prometheus(self) -> str:
        lines = []
        snap = self.snapshot()
        for c in snap["counters"]:
            lines.append(f"ensemble_{c['name']}{_labels(c['labels'])} {c['value']}")
        for h in snap["histograms"]:
            cum = 0
            for le, n in h["buckets_ms"].items():
                cum += n
                lines.append(f"ensemble_{h['name']}_ms_bucket{_labels({**h['labels'], 'le': le})} {cum}")
            lines.append(f"ensemble_{h['name']}_ms_sum{_labels(h['labels'])} {h['sum_ms']}")
            lines.append(f"ensemble_{h['name']}_ms_count{_labels(h['labels'])} {h['count']}")
        return "\n".join(lines) + "\n"


def _labels(d: dict) -> str:
    if not d:
        return ""
    return "{" + ",".join(f'{k}="{v}"' for k, v in sorted(d.items())) + "}"


class JsonFormatter(logging.Formatter):
    ALLOWED = ("request_id", "endpoint", "method", "status", "latency_ms", "bundle_version", "model_version",
               "catalog_version", "result_count", "fallback_levels", "personalized", "cache", "error_class",
               "customer_ref", "degraded", "event")

    def format(self, record: logging.LogRecord) -> str:
        body = {"ts": round(time.time(), 3), "level": record.levelname, "msg": record.getMessage()}
        for k in self.ALLOWED:
            if hasattr(record, k):
                body[k] = getattr(record, k)
        if record.exc_info:
            body["exception"] = self.formatException(record.exc_info)[-4000:]
        return json.dumps(body, default=str)


def request_logger() -> logging.Logger:
    log = logging.getLogger("ensemble.requests")
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(JsonFormatter())
        log.addHandler(h)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log

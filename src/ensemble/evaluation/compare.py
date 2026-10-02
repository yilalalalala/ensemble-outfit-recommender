"""Paired comparison of two Track A runs from their per-customer AP files.

  python -m ensemble.evaluation.compare <run_a> <run_b>

``run_*`` are names of files in ``data/interim/ap/`` without the extension, or
a backtest prefix (``backtest<tag>``), in which case every week present for
both runs is compared and the weeks are also pooled (customers within a week
are the resampling unit; weeks are stacked). Reports ΔMAP@12 = b − a with a
customer-level bootstrap 95% interval (``metrics.paired_bootstrap``).
"""
from __future__ import annotations

import json

import pandas as pd

from ensemble.config import load_config
from ensemble.evaluation.metrics import paired_bootstrap


def _files(d, name: str) -> dict[str, pd.DataFrame]:
    if (d / f"{name}.parquet").exists():
        return {"single": pd.read_parquet(d / f"{name}.parquet")}
    return {p.stem[len(name) + 1:]: pd.read_parquet(p) for p in sorted(d.glob(f"{name}_20*.parquet"))}


def compare(a: str, b: str) -> dict:
    d = load_config().path("interim") / "ap"
    fa, fb = _files(d, a), _files(d, b)
    weeks = sorted(set(fa) & set(fb))
    if not weeks:
        raise SystemExit(f"no common weeks for {a} and {b}")
    out, pooled = {}, []
    for w in weeks:
        m = fa[w].merge(fb[w], on="customer_idx", suffixes=("_a", "_b"))
        out[w] = paired_bootstrap(m.ap_a, m.ap_b)
        pooled.append(m.assign(week=w))
    if len(weeks) > 1:
        p = pd.concat(pooled)
        out["pooled"] = paired_bootstrap(p.ap_a, p.ap_b)
    return out


if __name__ == "__main__":
    import sys
    print(json.dumps(compare(sys.argv[1], sys.argv[2]), indent=1))

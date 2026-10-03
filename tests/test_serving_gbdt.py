"""The NumPy GBDT runtime must score exactly like LightGBM (it replaces it in the API process)."""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _in_subprocess(body: str) -> str:
    """LightGBM in a fresh interpreter: other test modules import torch (OpenMP clash on macOS)."""
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(body)], capture_output=True, text=True, timeout=600,
                       cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout


def test_numpy_forest_matches_lightgbm_exactly():
    out = _in_subprocess("""
        import json, tempfile
        import numpy as np, lightgbm as lgb
        from ensemble.serving.gbdt import Forest
        rng = np.random.default_rng(0)
        n = 6000
        X = np.c_[rng.normal(size=n), rng.normal(size=n), rng.integers(0, 12, n), rng.normal(size=n),
                  rng.integers(0, 4, n)].astype(np.float32)
        X[rng.random(n) < 0.2, 1] = np.nan           # NaN-missing feature
        X[rng.random(n) < 0.3, 3] = 0.0              # zeros (zero_as_missing below)
        y = (X[:, 0] + np.nan_to_num(X[:, 1]) + (X[:, 2] % 3 == 0) + rng.normal(size=n) > 1).astype(int)
        group = np.full(n // 20, 20)
        for zero_as_missing in (False, True):
            params = {"objective": "lambdarank", "num_leaves": 15, "min_child_samples": 5, "verbosity": -1,
                      "seed": 0, "deterministic": True, "force_col_wise": True, "zero_as_missing": zero_as_missing,
                      "max_cat_to_onehot": 2}
            ds = lgb.Dataset(X, label=y, group=group, categorical_feature=[2, 4],
                             feature_name=["a", "b", "c", "d", "e"])
            b = lgb.train(params, ds, num_boost_round=60)
            T = np.vstack([X[:2000], np.array([[np.nan, np.nan, -1, 0, np.nan], [0, 1, 99, np.nan, 7],
                                               [1e-40, -1e-40, np.nan, -0.0, 3.0]], np.float32)])
            ref = b.predict(T)
            f = Forest.from_booster_dump(b.dump_model())
            with tempfile.TemporaryDirectory() as d:
                f.save(d + "/m.json")
                got = Forest.load(d + "/m.json").predict(T)
            assert f.feature_names == ["a", "b", "c", "d", "e"]
            assert np.max(np.abs(got - ref)) < 1e-9, np.max(np.abs(got - ref))
            assert (np.argsort(-got, kind="stable") == np.argsort(-ref, kind="stable")).all()
            n_cat = sum(any(t["is_cat"]) for t in f.spec["trees"])
            print("ok", zero_as_missing, n_cat, float(np.max(np.abs(got - ref))))
    """)
    assert out.count("ok") == 2
    # Categorical splits must actually have been exercised.
    assert all(int(line.split()[2]) > 0 for line in out.splitlines() if line.startswith("ok"))

"""A dependency-free NumPy evaluator for LightGBM models (the serving-side ranker runtime).

Why: the API process loads PyTorch for visual search, and PyTorch and LightGBM ship clashing
OpenMP runtimes on macOS (a known crash in this repository). The ranker is therefore exported
once, offline, from ``Booster.dump_model()`` into flat per-tree arrays, and scored at request
time with NumPy only. This is the same idea as treelite / lleaves / m2cgen.

Exactness: the decision rules mirror LightGBM's ``Tree::NumericalDecision`` and
``Tree::CategoricalDecision`` (numerical: missing-value types None / Zero / NaN with
``default_left``, NaN under type None treated as 0; categorical: bitset membership, NaN and
negative values go right). Inputs are float32 cast to
float64 as LightGBM does. ``tests/test_serving.py`` asserts equality with ``Booster.predict``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

K_ZERO = 1e-35
MISSING = {"None": 0, "Zero": 1, "NaN": 2}


class Forest:
    """Flat arrays for every tree. Node ids: >= 0 internal, < 0 leaf (~leaf_index)."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.feature_names: list[str] = spec["feature_names"]
        self.trees = []
        for t in spec["trees"]:
            self.trees.append({k: (np.asarray(v) if k != "cat_sets" else [set(s) for s in v]) for k, v in t.items()})

    # -------------------------------------------------------------- export
    @staticmethod
    def from_booster_dump(dump: dict, num_iteration: int | None = None) -> "Forest":
        infos = dump["tree_info"][:num_iteration] if num_iteration else dump["tree_info"]
        trees = []
        for info in infos:
            nodes, leaves = [], []

            def walk(n):
                if "split_index" not in n:
                    leaves.append(float(n["leaf_value"]))
                    return -len(leaves)          # ~(len - 1) == -len
                idx = len(nodes)
                nodes.append(None)
                left, right = walk(n["left_child"]), walk(n["right_child"])
                is_cat = n["decision_type"] == "=="
                nodes[idx] = {
                    "feature": int(n["split_feature"]),
                    "threshold": 0.0 if is_cat else float(n["threshold"]),
                    "is_cat": is_cat,
                    "cats": [int(c) for c in str(n["threshold"]).split("||")] if is_cat else [],
                    "default_left": bool(n.get("default_left", False)),
                    "missing": MISSING[n.get("missing_type", "None")],
                    "left": left, "right": right}
                return idx

            root = walk(info["tree_structure"])
            if not nodes:                                   # a single-leaf tree
                trees.append({"feature": [], "threshold": [], "is_cat": [], "default_left": [], "missing": [],
                              "left": [], "right": [], "cat_sets": [], "leaf_value": leaves, "root": root})
                continue
            trees.append({"feature": [x["feature"] for x in nodes], "threshold": [x["threshold"] for x in nodes],
                          "is_cat": [x["is_cat"] for x in nodes], "default_left": [x["default_left"] for x in nodes],
                          "missing": [x["missing"] for x in nodes], "left": [x["left"] for x in nodes],
                          "right": [x["right"] for x in nodes], "cat_sets": [x["cats"] for x in nodes],
                          "leaf_value": leaves, "root": root})
        return Forest({"feature_names": list(dump["feature_names"]), "trees": trees,
                       "objective": dump.get("objective"), "num_trees": len(trees)})

    def save(self, path: Path) -> str:
        body = json.dumps(self.spec, separators=(",", ":"))
        Path(path).write_text(body)
        return hashlib.sha256(body.encode()).hexdigest()

    @staticmethod
    def load(path: Path) -> "Forest":
        return Forest(json.loads(Path(path).read_text()))

    # -------------------------------------------------------------- scoring
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Raw scores (sum of leaf values) for rows of ``X`` in ``feature_names`` order."""
        X = np.asarray(X, dtype=np.float32).astype(np.float64)
        n = len(X)
        out = np.zeros(n, dtype=np.float64)
        rows = np.arange(n)
        for t in self.trees:
            if len(t["feature"]) == 0:
                out += t["leaf_value"][~int(t["root"])]
                continue
            node = np.full(n, int(t["root"]), dtype=np.int64)
            active = node >= 0
            while active.any():
                idx = rows[active]
                nd = node[active]
                f = t["feature"][nd]
                v = X[idx, f]
                miss = t["missing"][nd]
                go_left = np.empty(len(idx), dtype=bool)
                cat = t["is_cat"][nd]
                num = ~cat
                if num.any():
                    vn = v[num].copy()
                    mn = miss[num]
                    nan = np.isnan(vn)
                    vn[nan & (mn != 2)] = 0.0
                    use_default = ((mn == 1) & (np.abs(vn) <= K_ZERO)) | ((mn == 2) & np.isnan(vn))
                    with np.errstate(invalid="ignore"):
                        gl = vn <= t["threshold"][nd[num]]
                    gl = np.where(use_default, t["default_left"][nd[num]], gl)
                    go_left[num] = gl
                if cat.any():
                    ci = np.flatnonzero(cat)
                    for j in ci:
                        x, node_id = v[j], nd[j]
                        # LightGBM 4 Tree::CategoricalDecision: NaN and negative values go right.
                        if np.isnan(x) or int(x) < 0:
                            go_left[j] = False
                            continue
                        go_left[j] = int(x) in t["cat_sets"][node_id]
                nxt = np.where(go_left, t["left"][nd], t["right"][nd])
                node[active] = nxt
                active = node >= 0
            out += t["leaf_value"][~node]
        return out

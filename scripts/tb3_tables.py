"""Render the Round-3 Track B report tables from the stored JSON artifacts.

  .venv/bin/python scripts/tb3_tables.py [report-name ...]

Every number in reports/TRACK_B_ROUND3_UPGRADE_REPORT.md comes out of here, so a
table can be regenerated from the artifacts instead of retyped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "reports" / "track_b_round3"


def load(name: str):
    p = D / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def pct(x) -> str:
    return "—" if x is None or x != x else f"{x:+.2%}"


def num(x, nd=4) -> str:
    return "—" if x is None or x != x else f"{x:.{nd}f}"


def systems_table(d, keys, labels, order=None) -> str:
    s = d["summary"]["systems"]
    names = order or list(s)
    head = "| system | " + " | ".join(labels) + " |"
    rule = "| --- | " + " | ".join("---:" for _ in labels) + " |"
    rows = []
    for n in names:
        if n not in s:
            continue
        cells = []
        for k in keys:
            v = s[n].get(k, {}).get("mean")
            cells.append(pct(v) if k.startswith("relative_lift") else num(v))
        rows.append(f"| `{n}` | " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *rows])


def per_fold_table(d, metric="recall@12", order=None) -> str:
    s = d["summary"]["systems"]
    folds = d["folds"]
    names = order or list(s)
    head = "| system | " + " | ".join(folds) + " | mean | between-fold SD | wins |"
    rule = "| --- | " + " | ".join("---:" for _ in folds) + " | ---: | ---: | ---: |"
    rows = []
    for n in names:
        if n not in s:
            continue
        m = s[n][metric]
        w = d["summary"]["wins_vs_shipped"].get(n)
        rows.append(f"| `{n}` | " + " | ".join(num(v) for v in m["per_fold"]) +
                    f" | **{num(m['mean'])}** | {num(m['std'])} | "
                    f"{'—' if w is None else f'{w}/{len(folds)}'} |")
    return "\n".join([head, rule, *rows])


def bootstrap_table(d, metric="recall@12", order=None) -> str:
    b = d["summary"]["pooled_bootstrap_vs_shipped"]
    names = order or list(b)
    head = "| system | Δ | 95% CI | relative | relative 95% CI | P(Δ>0) |"
    rule = "| --- | ---: | ---: | ---: | ---: | ---: |"
    rows = []
    for n in names:
        if n not in b:
            continue
        v = b[n][metric]
        rows.append(f"| `{n}` | {v['diff']:+.5f} | [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}] | "
                    f"{v['relative']:+.2%} | [{v['relative_ci95'][0]:+.2%}, "
                    f"{v['relative_ci95'][1]:+.2%}] | {v['p_gain_positive']:.3f} |")
    return "\n".join([head, rule, *rows])


def ablation_table(d, full="lgbm_personalized") -> str:
    s = d["summary"]["systems"]
    base = s[full]["recall@12"]["mean"]
    base_n = s[full]["ndcg@12"]["mean"]
    groups = (d["per_fold"][d["folds"][0]]["fit"].get("_feature_groups") or {}).get("removed_by_group", {})
    rows = []
    for n, m in s.items():
        if not n.startswith("lgbm_minus_") and n != "lgbm_compatibility":
            continue
        g = "personalization" if n == "lgbm_compatibility" else n[len("lgbm_minus_"):]
        r, nd = m["recall@12"]["mean"], m["ndcg@12"]["mean"]
        wins = sum(1 for a, b in zip(m["recall@12"]["per_fold"], base["per_fold"]) if a > b)
        rows.append((abs(r - base) / base, g, m, r, nd, wins))
    rows.sort(reverse=True)
    head = ("| feature group removed | features | Recall@12 | Δ vs full | NDCG@12 | Δ vs full | "
            "folds where removal wins |")
    rule = "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    out = [head, rule,
           f"| *(none — full model)* | {(d['per_fold'][d['folds'][0]]['fit'].get('_feature_groups') or {}).get('n_features', '—')}"
           f" | **{num(base['mean'])}** | — | **{num(base_n['mean'])}** | — | — |"]
    for _, g, m, r, nd, wins in rows:
        gl = g.replace("_", " + ") if "_" in g and g not in groups else g
        n_removed = sum(groups.get(x, 0) for x in g.split("_")) if g not in groups else groups.get(g, 0)
        out.append(f"| {gl} | {n_removed or '—'} | {num(r)} | {(r - base['mean']) / base['mean']:+.2%} | "
                   f"{num(nd)} | {(nd - base_n['mean']) / base_n['mean']:+.2%} | {wins}/{len(d['folds'])} |")
    return "\n".join(out)


def importance_table(d, model="lgbm_personalized", top=15) -> str:
    fits = [f["fit"][model] for f in d["per_fold"].values() if model in f["fit"]]
    agg: dict[str, list[float]] = {}
    for f in fits:
        for e in f["top_features"]:
            agg.setdefault(e["feature"], []).append(e["gain_share"])
    rows = sorted(((sum(v) / len(fits), k) for k, v in agg.items()), reverse=True)[:top]
    out = ["| feature | mean gain share | folds in top 25 |", "| --- | ---: | ---: |"]
    for v, k in rows:
        out.append(f"| `{k}` | {v:.2%} | {len(agg[k])}/{len(fits)} |")
    groups: dict[str, list[float]] = {}
    for f in fits:
        for g, v in (f.get("group_gain_share") or {}).items():
            groups.setdefault(g, []).append(v)
    out += ["", "| feature group | mean gain share |", "| --- | ---: |"]
    for g, v in sorted(groups.items(), key=lambda kv: -sum(kv[1])):
        out.append(f"| {g} | {sum(v) / len(v):.2%} |")
    return "\n".join(out)


def runtime_table(d) -> str:
    s, pf = d["summary"], d["per_fold"]
    rows = ["| quantity | value |", "| --- | ---: |",
            f"| folds | {s['n_folds']} |",
            f"| wall time | {s['total_seconds'] / 60:.0f} min |",
            f"| peak RSS | {s['peak_memory_mb'] / 1000:.1f} GB |",
            f"| candidates per query | {s['candidates_per_query']['mean']:.0f} |",
            f"| candidate-union recall (ceiling) | {s['union_recall']['mean']:.4f} |",
            f"| evaluated rows per fold | {sum(f['n_eval_rows'] for f in pf.values()) / len(pf):,.0f} |",
            f"| fit seconds per model per fold (mean) | "
            f"{sum(v['fit_seconds'] for f in pf.values() for k, v in f['fit'].items() if not k.startswith('_')) / sum(1 for f in pf.values() for k in f['fit'] if not k.startswith('_')):.0f} |",
            f"| scoring seconds per fold (all models) | {sum(f['eval_seconds'] for f in pf.values()) / len(pf):.0f} |"]
    return "\n".join(rows)


ORDER = ["slot_popularity", "assoc_npmi", "shipped_rrf_hybrid", "rrf_all_sources",
         "rrf_all_sources_union", "lgbm_compatibility", "lgbm_personalized"]
KEYS = ["recall@12", "recall@5", "ndcg@12", "relative_lift_recall@12_vs_shipped",
        "relative_lift_ndcg@12_vs_shipped", "relative_lift_recall@12_vs_popularity"]
LABELS = ["Recall@12", "Recall@5", "NDCG@12", "vs shipped (R@12)", "vs shipped (NDCG@12)", "vs popularity"]
SEG_KEYS = ["recall@12_tail", "recall@12_new_item", "recall@12_jewellery", "recall@12_returning",
            "recall@12_new_customer", "catalog_coverage@12", "novelty", "diversity_product_type@12",
            "diversity_product_code@12", "mean_list_length"]
SEG_LABELS = ["tail", "recently launched", "jewellery", "returning", "new customer", "coverage",
              "novelty", "div. type", "div. style", "list len"]


def main() -> None:
    names = sys.argv[1:] or ["backtest_val_ablations"]
    for name in names:
        d = load(name)
        if d is None:
            print(f"## {name}: not found\n")
            continue
        print(f"\n\n# {name}  (mode={d['mode']}, folds={', '.join(d['folds'])})\n")
        print("## headline\n")
        print(systems_table(d, KEYS, LABELS, ORDER))
        print("\n## per fold (Recall@12)\n")
        print(per_fold_table(d, "recall@12", ORDER))
        print("\n## per fold (NDCG@12)\n")
        print(per_fold_table(d, "ndcg@12", ORDER))
        print("\n## pooled customer-cluster bootstrap vs shipped_rrf_hybrid (Recall@12)\n")
        print(bootstrap_table(d, "recall@12", ORDER))
        print("\n## pooled customer-cluster bootstrap vs shipped_rrf_hybrid (NDCG@12)\n")
        print(bootstrap_table(d, "ndcg@12", ORDER))
        print("\n## beyond accuracy\n")
        print(systems_table(d, SEG_KEYS, SEG_LABELS, ORDER))
        print("\n## per slot (Recall@12)\n")
        slots = [k for k in d["summary"]["systems"][ORDER[0]] if k.startswith("recall@12_slot_")]
        print(systems_table(d, slots, [k.split("_slot_")[1] for k in slots], ORDER))
        if any(n.startswith("lgbm_minus_") for n in d["summary"]["systems"]):
            print("\n## feature-group ablation (full personalized model = baseline)\n")
            print(ablation_table(d))
        extra = [n for n in d["summary"]["systems"] if "+" in n]
        if extra:
            print("\n## serving re-ranking rules\n")
            print(systems_table(d, KEYS[:3] + SEG_KEYS[5:], LABELS[:3] + SEG_LABELS[5:],
                                ["lgbm_compatibility", *extra]))
        print("\n## feature importance (mean over folds)\n")
        print(importance_table(d))
        print("\n## runtime\n")
        print(runtime_table(d))


if __name__ == "__main__":
    main()

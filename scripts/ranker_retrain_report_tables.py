"""Render the tables of reports/TRACK_A_RANKER_RETRAIN_REPORT.md from
reports/track_a_retrain/comparison.json and shap_checks.json (written by `ranker_retrain report` / `shap`).

  .venv/bin/python scripts/ranker_retrain_report_tables.py            # print every table
  .venv/bin/python scripts/ranker_retrain_report_tables.py --render   # fill reports/track_a_retrain/report_template.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "reports" / "track_a_retrain"
C = json.loads((D / "comparison.json").read_text())
S = json.loads((D / "shap_checks.json").read_text()) if (D / "shap_checks.json").exists() else None
V = [v for v in ("R1", "R2", "R3") if v in C["variants"]]
FOLDS = list(C["r0"]["per_fold"])


def pct(x, d=2):
    return f"{x * 100:+.{d}f}%"


def ci(c):
    return f"[{c[0] * 100:+.2f}, {c[1] * 100:+.2f}]"


def rel(b):
    return f"{pct(b['relative'])} {ci(b['relative_ci95'])}"


def ladder() -> str:
    out = ["| variant | MAP@12 | vs R0 | Recall@12 vs R0 | NDCG@12 vs R0 | folds won | adopted by rule |",
           "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
           f"| R0 (fixed ranker) | {C['r0']['pooled']['map@12']:.5f} | — | — | — | — | baseline |"]
    for v in V:
        x = C["variants"][v]
        b = x["bootstrap_vs_r0"]
        out.append(f"| {v} | {x['pooled']['map@12']:.5f} | {rel(b['map@12'])} | {rel(b['recall@12'])} | "
                   f"{rel(b['ndcg@12'])} | {x['gates']['folds_won']}/6 | {'yes' if x['gates']['pass'] else 'no'} |")
    return "\n".join(out)


def per_fold() -> str:
    out = ["| fold | R0 | " + " | ".join(V) + " |", "| --- | ---: | " + " | ".join("---:" for _ in V) + " |"]
    for f in FOLDS:
        out.append(f"| {f} | {C['r0']['per_fold'][f]:.5f} | " +
                   " | ".join(f"{C['variants'][v]['per_fold'][f]['map@12']:.5f}" for v in V) + " |")
    return "\n".join(out)


def gates() -> str:
    dec = C["decision_rule"]
    rows = [("MAP@12 gain ≥ +1.0%", lambda g, x: (pct(g["map_gain"]), g["map_gain_ok"])),
            ("MAP@12 difference CI above 0", lambda g, x: (ci(x["bootstrap_vs_r0"]["map@12"]["relative_ci95"]),
                                                          g["map_ci_above_zero"])),
            (f"folds won ≥ {dec['min_folds_won']}/6", lambda g, x: (f"{g['folds_won']}/6", g["folds_ok"])),
            ("Recall@12 lower bound ≥ 0", lambda g, x: (ci(x["bootstrap_vs_r0"]["recall@12"]["relative_ci95"]),
                                                       g["recall12_lcb_ok"])),
            ("NDCG@12 lower bound ≥ 0", lambda g, x: (ci(x["bootstrap_vs_r0"]["ndcg@12"]["relative_ci95"]),
                                                     g["ndcg12_lcb_ok"])),
            ("segment lower bounds ≥ −2%", lambda g, x: (f"min {min(g['segment_lcbs'].values()) * 100:+.2f}% "
                                                        f"({min(g['segment_lcbs'], key=g['segment_lcbs'].get)})",
                                                        g["segments_ok"])),
            ("candidate recall identical", lambda g, x: (f"{x['pooled']['candidate_recall']:.4f}",
                                                        g["candidate_recall_identical"])),
            ("scoring time ≤ +20%", lambda g, x: (pct(x["cost"]["scoring_time_increase"], 1), None)),
            ("scoring RSS ≤ +25%", lambda g, x: (pct(x["cost"]["scoring_rss_increase"], 1), None)),
            ("training time ≤ 2×", lambda g, x: (f"{x['cost']['training_time_ratio']:.2f}×", g["cost_ok"]))]
    out = ["| gate | " + " | ".join(V) + " |", "| --- | " + " | ".join("---" for _ in V) + " |"]
    for name, fn in rows:
        cells = []
        for v in V:
            val, ok = fn(C["variants"][v]["gates"], C["variants"][v])
            cells.append(val if ok is None else f"{val} {'✓' if ok else '✗'}")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    out.append("| **all gates** | " + " | ".join("**pass**" if C["variants"][v]["gates"]["pass"] else "**fail**"
                                             for v in V) + " |")
    return "\n".join(out)


def segments() -> str:
    names = [("map@12_new", "new-customer MAP@12", "share_new"), ("map@12_returning", "returning MAP@12", "share_returning"),
             ("recall@12_tail", "tail-item Recall@12", "truth_share_tail"),
             ("recall@12_repeat", "repeat-purchase Recall@12", "truth_share_repeat"),
             ("recall@12_nonrepeat", "non-repeat Recall@12", "truth_share_nonrepeat"),
             ("recall@12_head", "head-item Recall@12", "truth_share_head"),
             ("recall@12_recent", "recent-item Recall@12", "truth_share_recent")]
    out = ["| segment | share | R0 | " + " | ".join(f"{v} vs R0" for v in V) + " |",
           "| --- | ---: | ---: | " + " | ".join("---:" for _ in V) + " |"]
    p0 = C["r0"]["pooled"]
    for k, label, share in names:
        out.append(f"| {label} | {p0[share] * 100:.1f}% | {p0[k]:.5f} | " +
                   " | ".join(rel(C["variants"][v]["bootstrap_vs_r0"][k]) for v in V) + " |")
    return "\n".join(out)


def costs() -> str:
    r0 = C["r0"]["cost"]
    out = ["| | R0 | " + " | ".join(V) + " |", "| --- | ---: | " + " | ".join("---:" for _ in V) + " |",
           f"| scoring time, six folds (s) | {r0['scoring_seconds']:.0f} | " +
           " | ".join(f"{C['variants'][v]['cost']['scoring_seconds']:.0f}" for v in V) + " |",
           f"| peak scoring RSS (GB) | {r0['scoring_peak_rss'] / 1e9:.2f} | " +
           " | ".join(f"{C['variants'][v]['cost']['scoring_peak_rss'] / 1e9:.2f}" for v in V) + " |",
           f"| training time per fold, mean (s) | {C['decision_rule']['baseline_training_seconds']} | " +
           " | ".join(f"{C['variants'][v]['cost']['training_seconds_mean']:.0f}" for v in V) + " |",
           "| peak training RSS (GB) | — | " +
           " | ".join(f"{C['variants'][v]['cost']['training_peak_rss'] / 1e9:.2f}" for v in V) + " |",
           "| training rows per fold, mean | — | " +
           " | ".join(f"{C['variants'][v]['cost']['train_rows_mean'] / 1e6:.2f}M" for v in V) + " |",
           "| trees, mean | — | " + " | ".join(f"{C['variants'][v]['cost']['trees_mean']:.0f}" for v in V) + " |"]
    m = C["matrices"]
    tr = [x for k, x in m.items() if k.startswith("train_")]
    ev = [x for k, x in m.items() if k.startswith("eval_")]
    out.append(f"\nShared, built once for all variants: {len(tr)} training matrices "
               f"({min(x['rows'] for x in tr) / 1e6:.1f}–{max(x['rows'] for x in tr) / 1e6:.1f}M rows after the training-row "
               f"rule, {sum(x['seconds'] for x in tr):.0f} s in total, peak {max(x['peak_rss_bytes'] for x in tr) / 1e9:.1f} GB) "
               f"and {len(ev)} evaluation matrices ({min(x['rows'] for x in ev) / 1e6:.1f}–{max(x['rows'] for x in ev) / 1e6:.1f}M "
               f"rows, {sum(x['seconds'] for x in ev):.0f} s, peak {max(x['peak_rss_bytes'] for x in ev) / 1e9:.1f} GB).")
    return "\n".join(out)


def importance() -> str:
    tops = {v: list(C["variants"][v]["importance_top"].items())[:10] for v in V}
    out = ["| # | " + " | ".join(V) + " |", "| ---: | " + " | ".join("---" for _ in V) + " |"]
    for i in range(10):
        out.append(f"| {i + 1} | " + " | ".join(f"`{tops[v][i][0]}` {tops[v][i][1] * 100:.1f}%" for v in V) + " |")
    if "R2" in V:
        ps = C["variants"]["R2"]["prov_gain_share"]
        out.append("\nR2 provenance gain share (mean over folds): " +
                   ", ".join(f"`{k}` {x * 100:.2f}%" for k, x in ps.items()) +
                   f"; total {sum(ps.values()) * 100:.2f}%.")
    return "\n".join(out)


def shap() -> str:
    if S is None:
        return "*(SHAP checks not yet run)*"
    out = [f"Sample: the top-{S['k']} list of the first {S['n_customers']:,} customers (by id) of fold {S['fold']}'s "
           f"first evaluation part, seed {S['seed']} models.\n",
           "| | " + " | ".join(S["variants"]) + " |", "| --- | " + " | ".join("---:" for _ in S["variants"]) + " |"]
    rows = [("rows explained", lambda x: f"{x['rows']:,}"),
            ("additivity, max abs error", lambda x: f"{x['additivity_max_abs_err']:.1e}"),
            ("top-12 rows that are SASRec-appended", lambda x: f"{x['top_k_share_appended'] * 100:.1f}%"),
            ("provenance share of mean absolute SHAP", lambda x: f"{x['prov_share_total'] * 100:.2f}%"),
            ("rows with a provenance feature in the top-3 absolute SHAP", lambda x: f"{x['rows_with_prov_in_top3_shap']:,}"),
            ("reason chips shown", lambda x: f"{x['chips']:,}"),
            ("rows without any chip", lambda x: f"{x['rows_without_chip']:,}"),
            ("unsupported chips (evidence-gated, D-030)", lambda x: f"{x['unsupported_chips']}"),
            ("unsupported chips if the gate were off", lambda x: f"{x['unsupported_chips_without_gate']:,}")]
    for name, fn in rows:
        out.append(f"| {name} | " + " | ".join(fn(x) for x in S["variants"].values()) + " |")
    return "\n".join(out)


def seeds() -> str:
    if not C["seeds"]:
        return "*(seed replicates not yet run)*"
    out = []
    for v, s in C["seeds"].items():
        out.append(f"| {v} seed | MAP@12 | vs R0 | " + " | ".join(FOLDS) + " |")
        out.append("| --- | ---: | ---: | " + " | ".join("---:" for _ in FOLDS) + " |")
        for sd, m in s["per_seed_map@12"].items():
            out.append(f"| {sd} | {m:.5f} | {pct(s['map_rel_gain_vs_r0'][sd])} | " +
                       " | ".join(f"{s['per_seed_per_fold'][sd][f]:.5f}" for f in FOLDS) + " |")
        out.append(f"\n{v}: mean {s['mean']:.5f}, SD {s['sd']:.5f} over {len(s['per_seed_map@12'])} seeds; "
                   f"mean gain vs R0 {pct(sum(s['map_rel_gain_vs_r0'].values()) / len(s['map_rel_gain_vs_r0']))}.")
    return "\n".join(out)


TABLES = {"LADDER": ladder, "PER_FOLD": per_fold, "GATES": gates, "SEGMENTS": segments, "COSTS": costs,
          "IMPORTANCE": importance, "SHAP": shap, "SEEDS": seeds}


if __name__ == "__main__":
    if "--render" in sys.argv:
        t = (D / "report_template.md").read_text()
        for k, fn in TABLES.items():
            t = t.replace("{{" + k + "}}", fn())
        assert "{{" not in t, "unfilled placeholder"
        (ROOT / "reports" / "TRACK_A_RANKER_RETRAIN_REPORT.md").write_text(t)
        print("rendered reports/TRACK_A_RANKER_RETRAIN_REPORT.md")
    else:
        for k, fn in TABLES.items():
            print(f"## {k}\n\n{fn()}\n")

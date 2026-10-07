"""Render the tables of reports/TRACK_A_RETRIEVAL_UPGRADE_REPORT.md from
reports/track_a_retrieval/comparison.json (written by `retrieval_upgrade report`).

  .venv/bin/python scripts/retrieval_report_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C = json.loads((ROOT / "reports" / "track_a_retrieval" / "comparison.json").read_text())
CH = ["repeat", "pop", "pop_age", "cf", "variant", "dept_pop", "section_pop", "covis", "new_arrival_pers"]


def pct(x, d=2):
    return f"{x * 100:+.{d}f}%"


def ci(c):
    return f"[{c[0] * 100:+.2f}, {c[1] * 100:+.2f}]"


def diagnostics() -> str:
    d = C["diagnostics"]
    out = ["### Channels in the current union (pooled over six folds; recall = share of all purchased pairs)\n",
           "| channel | cap | recall alone | unique | in pool beyond cap |", "| --- | ---: | ---: | ---: | ---: |"]
    caps = C["configs"]["final"]["spec"]["caps"]
    for c in CH:
        out.append(f"| {c} | {caps.get(c, 0)} | {d['channel_recall_alone'][c]:.4f} | {d['channel_unique_recall'][c]:.4f} | "
                   f"{d['channel_beyond_cap_recall'][c]:.4f} |")
    out.append(f"\nRedundant candidate rate (rows removed by de-duplication): {d['redundant_candidate_rate']:.3f}.")
    out.append("\n### Miss taxonomy (pooled share of all purchased pairs, mutually exclusive)\n")
    out.append("| outcome | share |")
    out.append("| --- | ---: |")
    for k, v in d["miss_taxonomy"].items():
        if k == "no_channel_incl_neural_top1000":
            continue
        out.append(f"| {k.replace('_', ' ')} | {v:.4f} |")
    out.append(f"\nOf the eligible pairs no current channel proposes, the share also absent from all three "
               f"neural top-1000 lists is {d['miss_taxonomy']['no_channel_incl_neural_top1000']:.4f} of all pairs.")
    out.append("\n### Marginal recall of each neural list appended to the current union (pooled)\n")
    out.append("| list | top-100 | top-300 | top-1000 |")
    out.append("| --- | ---: | ---: | ---: |")
    for m in ("sasrec", "lightgcn", "bpr"):
        nm = d["neural_marginal_recall"]
        out.append(f"| {m} | {nm[f'{m}@100']:.4f} | {nm[f'{m}@300']:.4f} | {nm[f'{m}@1000']:.4f} |")
    out.append("\n### Union recall by segment (current union, pooled)\n")
    out.append("| segment | candidate recall |")
    out.append("| --- | ---: |")
    for s, v in d["segments_union_recall"].items():
        out.append(f"| {s} | {v['recall']:.4f} |")
    j = d["pairwise_jaccard_mean_over_folds"]
    top = sorted(j.items(), key=lambda x: -x[1])[:5]
    out.append("\nHighest pairwise Jaccard overlaps: " + ", ".join(f"{k} {v:.3f}" for k, v in top) + ".")
    r = d["reconcile"]
    out.append(f"\n**Reconciliation.** Recomputed candidate hits {r['recomputed_hits']:,} = stored {r['stored_hits']:,} over "
               f"{r['truth_pairs']:,} purchased pairs → candidate recall {r['candidate_recall']:.4f}; "
               f"{r['customers_with_n_cand_difference']} customers differ in candidate *count* by a few rows "
               f"(max relative difference {r['max_abs_n_cand_relative_difference']:.1e}).")
    return "\n".join(out)


def configs() -> str:
    cf = C["configs"]
    order = ["final", "sasrec_A300", "lightgcn_A300", "bpr_A300", "neural3_A300", "F300",
             "sasrec_A500", "lightgcn_A500", "bpr_A500", "neural3_A500", "F500",
             "sasrec_A1000", "lightgcn_A1000", "bpr_A1000", "neural3_A1000", "F1000"]
    out = ["### All configurations, six folds pooled (fixed seed-42 ranker per fold)\n",
           "| config | cand. mean / p50 / p95 | candidate recall | Recall@12 | MAP@12 | NDCG@12 | marginal conversion | rank+score s (×final) | peak RSS GB |",
           "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for n in [x for x in order if x in cf]:
        c = cf[n]
        q, k, co = c["pooled"], c["candidates_per_customer"], c["cost"]
        mc = c.get("marginal_conversion")
        out.append(f"| {n} | {k['mean']:.0f} / {k['p50']:.0f} / {k['p95']:.0f} | {q['candidate_recall']:.4f} | "
                   f"{q['recall@12']:.4f} | {q['map@12']:.5f} | {q['ndcg@12']:.5f} | "
                   f"{'—' if mc is None else f'{mc:.3f}'} | {co['rank_score_seconds']:.0f} ({co['time_ratio_vs_final']:.2f}×) | "
                   f"{co['peak_rss_bytes'] / 1e9:.2f} |")
    out.append("\n### Bootstrap versus the current final system (relative change, 95% CI; customers across folds)\n")
    out.append("| config | candidate recall | MAP@12 | Recall@12 | NDCG@12 | tail cand. recall | new-cust. cand. recall | tail Recall@12 | new-cust. MAP@12 |")
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for n in [x for x in order if x in cf and x != "final"]:
        b = cf[n]["bootstrap_vs_final"]
        cell = lambda k: f"{pct(b[k]['relative'])} {ci(b[k]['relative_ci95'])}"  # noqa: E731
        out.append(f"| {n} | {cell('candidate_recall')} | {cell('map@12')} | {cell('recall@12')} | {cell('ndcg@12')} | "
                   f"{cell('tail_candidate_recall')} | {cell('new_customer_candidate_recall')} | {cell('tail_recall@12')} | "
                   f"{cell('new_customer_map@12')} |")
    out.append("\n### The predeclared adoption rule, applied mechanically\n")
    out.append("| config | cand. recall ≥ 0.2021 | MAP CI > 0 | R@12 CI > 0 | folds gain / worst fold | tail / new LCB | time ×, RSS | pass |")
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for n in [x for x in order if x in cf and x != "final"]:
        g = cf[n]["decision_gates"]
        yn = lambda v: "yes" if v else "no"  # noqa: E731
        out.append(f"| {n} | {yn(g['candidate_recall_target'])} | {yn(g['map_ci_above_zero'])} | "
                   f"{yn(g['recall12_ci_above_zero'])} | {g['folds_with_candidate_gain']}/6, {pct(g['worst_fold_candidate_rel'])} | "
                   f"{pct(g['tail_candidate_lcb'])} / {pct(g['new_customer_candidate_lcb'])} | "
                   f"{g['time_ratio']:.2f}×, {pct(g['rss_increase'], 0)} | **{yn(g['pass'])}** |")
    out.append("\n### Per fold (candidate recall / MAP@12)\n")
    folds = list(cf["final"]["per_fold"])
    out.append("| config | " + " | ".join(f[5:] for f in folds) + " |")
    out.append("| --- | " + " | ".join("---" for _ in folds) + " |")
    for n in [x for x in order if x in cf]:
        pf = cf[n]["per_fold"]
        out.append(f"| {n} | " + " | ".join(f"{pf[f]['candidate_recall']:.3f} / {pf[f]['map@12']:.5f}" for f in folds) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    print("## Diagnostics tables\n")
    print(diagnostics())
    print("\n## Configuration tables\n")
    print(configs())

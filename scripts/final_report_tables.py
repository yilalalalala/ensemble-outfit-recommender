"""Render the tables of reports/TRACK_A_RESEARCH_TRACK_B_PRODUCTION_REPORT.md from stored artifacts.

  .venv/bin/python scripts/final_report_tables.py > /tmp/tables.md

Every number in the report's tables comes from here; scripts/verify_final_report.py re-reads the
same artifacts and asserts the figures quoted in the prose.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
NAMES = {"popularity_global": "Global popularity", "popularity_age": "Age-band popularity",
         "repeat_pop_age": "Repeat purchase + age-band popularity", "item_cf": "Item-to-item CF (cosine)",
         "covis": "Co-visitation", "bpr": "BPR-MF", "lightgcn": "LightGCN", "sasrec": "SASRec (gBCE)",
         "ens_phase2": "Phase-2 ensemble (reproduced)", "ens_phase2_nscores": "**Final: Phase-2 + neural scores**",
         "ens_realloc": "Re-allocated candidates (control)", "ens_neural_channels": "Neural channels + scores"}
ORDER = list(NAMES)


def f5(x):
    return f"{x:.5f}"


def pct(x, d=2):
    return f"{x * 100:+.{d}f}%"


def ci(c):
    return f"[{c[0] * 100:+.2f}, {c[1] * 100:+.2f}]"


def load(p):
    return json.loads((R / p).read_text())


def track_a() -> str:
    c = load("track_a_research/comparison.json")
    S = c["systems"]
    out = []
    folds = list(next(iter(S.values()))["seeds"].values())[0]["per_fold"]
    out.append("### Pooled six-fold results (primary seed; ± = SD over seeds where run)\n")
    out.append("| system | MAP@12 | ± seeds | NDCG@12 | Recall@12 | coverage | novelty | seeds |")
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for s in [x for x in ORDER if x in S]:
        b = S[s]
        prim = b["seeds"].get("42") or b["seeds"].get("0")
        p = prim["pooled"]
        ss = b["seed_stats"]["map@12"]
        out.append(f"| {NAMES[s]} | {f5(ss['mean'])} | {f5(ss['std']) if ss['n_seeds'] > 1 else '—'} | "
                   f"{f5(p['ndcg@12'])} | {f5(p['recall@12'])} | {p['coverage_mean']:.3f} | "
                   f"{p['novelty_mean']:.2f} | {ss['n_seeds']} |")
    out.append("\n### Per fold (MAP@12, mean over seeds)\n")
    out.append("| system | " + " | ".join(f[5:] for f in folds) + " |")
    out.append("| --- | " + " | ".join("---:" for _ in folds) + " |")
    for s in [x for x in ORDER if x in S]:
        pf = S[s]["per_fold_seed_mean"]
        out.append(f"| {NAMES[s]} | " + " | ".join(f5(pf[f]["mean"]) for f in folds) + " |")
    out.append("\n### Paired customer-cluster bootstrap (MAP@12; 1,000 resamples, customers across folds)\n")
    out.append("| a → b | MAP@12 a | MAP@12 b | relative | 95% CI (%) | NDCG@12 rel. | Recall@12 rel. | folds won by b |")
    out.append("| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |")
    for k in c["comparisons"]:
        m = k["map@12"]
        out.append(f"| {NAMES.get(k['a'], k['a'])} → {NAMES.get(k['b'], k['b'])} | {f5(m['mean_a'])} | {f5(m['mean_b'])} | "
                   f"{pct(m['relative'])} | {ci(m['relative_ci95'])} | {pct(k['ndcg@12']['relative'])} | "
                   f"{pct(k['recall@12']['relative'])} | {k['folds_won_by_b']}/6 |")
    final = c["final_system"]
    p = S[final]["seeds"]["42"]["pooled"]
    out.append("\n### Segments (pooled, primary seed)\n")
    keys = [("map@12_returning", "MAP@12 returning"), ("map@12_new", "MAP@12 new customers"),
            ("map@12_activity_low", "MAP@12 low activity"), ("map@12_activity_medium", "MAP@12 medium"),
            ("map@12_activity_high", "MAP@12 high"), ("recall@12_head", "Recall@12 head items"),
            ("recall@12_tail", "Recall@12 tail items"), ("recall@12_recent", "Recall@12 recently launched"),
            ("recall@12_repeat", "Recall@12 repeat truth"), ("recall@12_nonrepeat", "Recall@12 non-repeat truth")]
    show = [s for s in ("repeat_pop_age", "sasrec", "ens_phase2", final) if s in S]
    out.append("| segment | " + " | ".join(NAMES[s] for s in show) + " |")
    out.append("| --- | " + " | ".join("---:" for _ in show) + " |")
    for k, label in keys:
        out.append(f"| {label} | " + " | ".join(
            f5((S[s]["seeds"].get("42") or S[s]["seeds"]["0"])["pooled"][k]) for s in show) + " |")
    out.append(f"\nTruth shares: returning customers {p['share_returning']:.3f}, ineligible (no sale in 28 days) "
               f"{p['truth_share_ineligible']:.3f}, cold {p['truth_share_cold']:.3f}, recently launched "
               f"{p['truth_share_recent']:.3f}, repeat {p['truth_share_repeat']:.3f}, head {p['truth_share_head']:.3f}.")
    out.append("\n### Candidate recall ceiling and conversion (primary seed)\n")
    out.append("| system | candidates / customer | candidate recall | Recall@12 | conversion |")
    out.append("| --- | ---: | ---: | ---: | ---: |")
    for s in [x for x in ORDER if x.startswith("ens_") and x in S]:
        q = S[s]["seeds"]["42"]["pooled"]
        out.append(f"| {NAMES[s]} | {q['candidates_per_customer']:.1f} | {q['candidate_recall']:.4f} | "
                   f"{q['recall@12']:.4f} | {q['conversion']:.3f} |")
    out.append("\n### Resource cost (mean per fold)\n")
    out.append("| system | train s | score s | peak RSS GB | parameters / trees | artifact MB | device |")
    out.append("| --- | ---: | ---: | ---: | --- | ---: | --- |")
    for s in ("bpr", "lightgcn", "sasrec"):
        if s in S:
            r = S[s]["seeds"]["0"]["resources"]
            out.append(f"| {NAMES[s]} | {r['train_seconds']:.0f} | {r['score_seconds']:.0f} | "
                       f"{r['peak_rss_bytes'] / 1e9:.2f} | {r['n_parameters'] / 1e6:.1f} M params | "
                       f"{r['artifact_bytes'] / 1e6:.0f} | {r['device']} |")
    for s in [x for x in ORDER if x.startswith("ens_") and x in S]:
        r = S[s]["seeds"]["42"]["resources"]
        out.append(f"| {NAMES[s]} | {r['fit_seconds'] + r['load_seconds']:.0f} | {r['score_seconds']:.0f} | "
                   f"{r['peak_rss_bytes'] / 1e9:.2f} | {r['n_trees']:.0f} trees, {r['n_features']:.0f} features | "
                   f"{r['model_bytes'] / 1e6:.1f} | cpu |")
    if c.get("ablations"):
        out.append("\n### Ablations of the final system (each refitted on all six folds, seed 42)\n")
        out.append("| removed | MAP@12 | relative to final | 95% CI (%) | folds where removal wins | features |")
        out.append("| --- | ---: | ---: | --- | ---: | ---: |")
        for a, v in c["ablations"].items():
            m = v["map@12"]
            out.append(f"| {a} | {f5(m['mean_b'])} | {pct(m['relative'])} | {ci(m['relative_ci95'])} | "
                       f"{v['folds_ablation_wins']}/6 | {v['resources']['n_features']:.0f} |")
    nc = c.get("neural_retrieval_coverage") or {}
    if nc:
        out.append("\n### Neural models as retrievers (seed 0, mean over folds)\n")
        out.append("| model | Recall@12 | Recall@50 | Recall@100 | buyers represented | distinct articles in top-12 |")
        out.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for m, per in nc.items():
            v = list(per.values())
            mean = lambda k: sum(x[k] for x in v) / len(v)  # noqa: E731
            out.append(f"| {NAMES[m]} | {mean('recall@12'):.4f} | {mean('recall@50'):.4f} | {mean('recall@100'):.4f} | "
                       f"{mean('buyers_represented'):.3f} | {mean('distinct_top12'):,.0f} |")
    rep = c.get("phase2_reproduction") or {}
    if rep.get("weeks"):
        out.append("\n### Reproduction of the authoritative Phase-2 backtest\n")
        out.append("| week | Phase-2 backtest | this protocol | relative |")
        out.append("| --- | ---: | ---: | ---: |")
        for w, v in rep["weeks"].items():
            out.append(f"| {w} | {f5(v['phase2_backtest'])} | {f5(v['this_protocol'])} | {pct(v['relative'])} |")
    return "\n".join(out)


def track_b() -> str:
    out = []
    sr = load("track_b_production/serving_regression.json")
    out.append("### Serving regression on six rolling folds (D-041)\n")
    out.append("| comparison | Recall@12 a | Recall@12 b | relative | Δ 95% CI | NDCG@12 rel. |")
    out.append("| --- | ---: | ---: | ---: | --- | ---: |")
    for k in sr["comparisons"]:
        r, n = k["recall@12"], k["ndcg@12"]
        out.append(f"| {k['a']} → {k['b']} | {r['mean_a']:.4f} | {r['mean_b']:.4f} | {pct(r['mean_b'] / r['mean_a'] - 1)} | "
                   f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | {pct(n['mean_b'] / n['mean_a'] - 1)} |")
    m = load("track_b_production/bundle_manifest.json")
    out.append("\n### Serving bundle\n")
    out.append(f"Version `{m['bundle_version']}`, serving week {m['serving_week']} (cutoff {m['cutoff']}), "
               f"{m['n_keys']:,} keys, {m['n_pool_rows']:,} pool rows (P = {m['pool']}), "
               f"{m['n_live_articles']:,} live articles, {m['profiles']['n_customers']:,} customer profiles, "
               f"equivalence {m['equivalence']['identical_order']}/{m['equivalence']['requests']} identical orderings.")
    for f in sorted((R / "track_b_production").glob("bench_*.json")):
        b = json.loads(f.read_text())
        out.append(f"\n### Benchmark `{f.stem}` (bundle {b['bundle_total_mb']:,.0f} MB)\n")
        out.append("| mode | concurrency | p50 ms | p95 ms | p99 ms | throughput rps | errors |")
        out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for mode in ("cold", "warm"):
            if mode not in b:
                continue
            for cc in (1, 4, 16):
                x = b[mode][f"c{cc}"]
                out.append(f"| {mode} | {cc} | {x['p50_ms']:.1f} | {x['p95_ms']:.1f} | {x['p99_ms']:.1f} | "
                           f"{x['throughput_rps']:.0f} | {x['errors'] + x['timeouts']} |")
        for mode in ("cold", "warm"):
            if mode in b:
                x = b[mode]
                out.append(f"\n{mode}: startup→ready {x['startup_to_ready_s']:.2f} s, first request "
                           f"{x['first_request_ms']:.0f} ms, RSS after ready {x['rss_after_ready_mb']:.0f} MB, "
                           f"steady {x['rss_steady_mb']:.0f} MB, peak {x['rss_peak_mb']:.0f} MB, cache {x.get('cache')}")
        v = b.get("warm", {}).get("visual")
        if v and "first_call_ms" in v:
            out.append(f"\nvisual search ({v.get('mode')}): first call {v['first_call_ms']:.0f} ms; warm c1 p50 "
                       f"{v['warm_c1']['p50_ms']:.0f} / p95 {v['warm_c1']['p95_ms']:.0f} ms; c4 p50 "
                       f"{v['warm_c4']['p50_ms']:.0f} / p95 {v['warm_c4']['p95_ms']:.0f} ms")
        e = b.get("exact_vector_search")
        if e:
            out.append(f"\nexact top-8 over {e['n_vectors']:,} × {e['dim']} vectors: {e['exact_top8_ms_per_query']:.2f} ms/query "
                       f"({e['index_mb']:.0f} MB)")
    return "\n".join(out)


if __name__ == "__main__":
    print("## Track A tables\n")
    print(track_a())
    print("\n## Track B tables\n")
    print(track_b())

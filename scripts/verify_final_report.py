"""Assert the headline figures of reports/TRACK_A_RESEARCH_TRACK_B_PRODUCTION_REPORT.md.

  .venv/bin/python scripts/verify_final_report.py

Each check recomputes a figure from a stored artifact, formats it exactly as the report prints it,
and asserts that string is in the report. Exits non-zero listing every mismatch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
REPORT = R / "TRACK_A_RESEARCH_TRACK_B_PRODUCTION_REPORT.md"


def j(p):
    return json.loads((R / p).read_text())


def pct(x, d=2):
    return f"{x * 100:+.{d}f}%"


def checks() -> list[tuple[str, str]]:
    c = j("track_a_research/comparison.json")
    S = c["systems"]
    seg = j("track_a_research/segments_and_seeds.json")
    cmp_ = {(k["a"], k["b"]): k for k in c["comparisons"]}
    final = c["final_system"]
    out = []
    # --- Track A pooled MAP@12 (three-seed means for seeded systems)
    for s, label in (("popularity_global", "global popularity"), ("repeat_pop_age", "repeat purchase + age-band")):
        out.append((label, f"{S[s]['seed_stats']['map@12']['mean']:.5f}"))
    for s in ("bpr", "lightgcn", "sasrec", "ens_phase2", final):
        out.append((f"{s} 3-seed MAP@12", f"{S[s]['seed_stats']['map@12']['mean']:.5f}"))
        assert S[s]["seed_stats"]["map@12"]["n_seeds"] == 3, s
    out.append(("BPR seed SD", f"± {S['bpr']['seed_stats']['map@12']['std']:.5f}"))
    out.append(("LightGCN seed SD", f"± {S['lightgcn']['seed_stats']['map@12']['std']:.5f}"))
    out.append(("SASRec seed SD", f"± {S['sasrec']['seed_stats']['map@12']['std']:.5f}"))
    assert c["strongest_reproduced_baseline"] == "sasrec" and c["strongest_baseline_any"] == "repeat_pop_age"
    assert final == "ens_phase2_nscores"
    # --- key comparisons
    for a, b in (("sasrec", final), ("repeat_pop_age", final), ("ens_phase2", final)):
        m = cmp_[(a, b)]["map@12"]
        out.append((f"{a}->{b} relative", pct(m["relative"])))
        out.append((f"{a}->{b} CI", f"[{pct(m['relative_ci95'][0])}, {pct(m['relative_ci95'][1])}]"))
        assert cmp_[(a, b)]["folds_won_by_b"] == 6
    out.append(("channels vs final", pct(cmp_[(final, "ens_neural_channels")]["map@12"]["relative"])))
    out.append(("channels vs phase2", pct(cmp_[("ens_phase2", "ens_neural_channels")]["map@12"]["relative"])))
    out.append(("realloc vs phase2", pct(cmp_[("ens_phase2", "ens_realloc")]["map@12"]["relative"])))
    for x in (-0.5777, -0.3558):     # baselines vs repeat rule quoted as a range −36% … −58%
        pass
    assert round(cmp_[("repeat_pop_age", "sasrec")]["map@12"]["relative"] * 100) == -36
    assert round(cmp_[("repeat_pop_age", "bpr")]["map@12"]["relative"] * 100) == -58
    # adoption decisions
    d = c["adoption_decisions"]
    assert d[final]["adopt"] and not d["ens_neural_channels"]["adopt"] and not d["ens_realloc"]["adopt"]
    out.append(("channels time increase", f"+{d['ens_neural_channels']['train_time_increase'] * 100:.0f}%"))
    out.append(("realloc time increase", f"+{d['ens_realloc']['train_time_increase'] * 100:.2f}%"))
    # per-seed relative gains
    for k in ("42", "43", "44"):
        out.append((f"seed {k} gain", pct(seg["per_seed_relative_final_vs_phase2"][k])))
    for s, label in (("ens_phase2", "phase2"), (final, "final")):
        for k in ("42", "43", "44"):
            out.append((f"{label} seed {k}", f"{seg['per_seed_pooled_map@12'][s][k]:.5f}"))
    # segment trade-offs
    st = seg["segment_tradeoffs_vs_phase2"]
    for k in ("recall@12_recent", "recall@12_repeat", "recall@12_tail"):
        out.append((k, pct(st[k]["relative"])))
        out.append((k + " CI", f"[{st[k]['relative_ci95'][0] * 100:+.2f}, {st[k]['relative_ci95'][1] * 100:+.2f}]"))
    out.append(("nonrepeat", pct(st["recall@12_nonrepeat"]["relative"])))
    out.append(("head", pct(st["recall@12_head"]["relative"])))
    out.append(("new customers", pct(st["map@12_new_customers"]["relative"])))
    out.append(("returning", pct(st["map@12_returning"]["relative"])))
    out.append(("recent short", f"+{st['recall@12_recent']['relative'] * 100:.1f}%"))
    out.append(("nonrepeat short", f"+{st['recall@12_nonrepeat']['relative'] * 100:.1f}%"))
    out.append(("tail short", f"{st['recall@12_tail']['relative'] * 100:.1f}%"))
    out.append(("repeat short", f"{st['recall@12_repeat']['relative'] * 100:.1f}%"))
    # ablations
    for a, v in c["ablations"].items():
        out.append((f"ablation {a}", pct(v["map@12"]["relative"])))
        assert v["map@12"]["relative_ci95"][1] < 0, a
    # failure analysis
    fa = c["failure_analysis"]
    out.append(("hits", f"{fa['pair_modes']['hit_in_top12'] * 100:.2f}%"))
    out.append(("retrieved not shown", f"{fa['pair_modes']['retrieved_not_in_top12'] * 100:.2f}%"))
    out.append(("not retrieved", f"{fa['pair_modes']['not_retrieved'] * 100:.2f}%"))
    out.append(("rows", f"{fa['n_customers_rows']:,}"))
    assert len(fa["examples"]) >= 20
    for mode, n in fa["example_modes"].items():
        out.append((f"example mode {mode}", f"({n})"))
    p = S[final]["seeds"]["42"]["pooled"]
    out.append(("candidate recall", f"{p['candidate_recall'] * 100:.1f}%"))
    out.append(("conversion", f"{p['conversion'] * 100:.1f}%"))
    # reproduction
    rep = c["phase2_reproduction"]["weeks"]
    lo = min(abs(v["relative"]) for v in rep.values())
    hi = max(abs(v["relative"]) for v in rep.values())
    out.append(("reproduction range", f"{lo * 100:.2f}%–{hi * 100:.2f}%"))
    # CPU reproducibility
    cr = j("track_a_research/cpu_reproducibility.json")
    assert cr["user_emb_max_abs_diff"] == 0 and cr["item_emb_max_abs_diff"] == 0 and cr["topk_identical"]
    # tuning
    tu = j("track_a_research/tuning.json")
    sel = tu["selected"]
    assert (sel["bpr"]["dim"], sel["bpr"]["epochs"], sel["lightgcn"]["lr"], sel["lightgcn"]["epochs"],
            sel["sasrec"]["epochs"], sel["sasrec"]["loss"]) == (128, 24, 0.02, 27, 24, "gbce")
    a = {r["overrides"].get("loss"): r["best"]["raw_map@12"] for r in tu["stage_a"]["sasrec"]}
    out.append(("gbce vs bce", f"+{(a['gbce'] / a['bce'] - 1) * 100:.0f}%"))
    # --- Track B
    sr = j("track_b_production/serving_regression.json")
    comp = {(k["a"], k["b"]): k for k in sr["comparisons"]}
    k = comp[("lgbm_compatibility+shipped", "lgbm_personalized@pool100+shipped")]["recall@12"]
    out.append(("served gain", pct(k["mean_b"] / k["mean_a"] - 1)))
    out.append(("served CI", f"[{k['ci95'][0]:+.4f}, {k['ci95'][1]:+.4f}]"))
    out.append(("retention P100", f"{sr['d041']['retention_vs_unrestricted']['100'] * 100:.1f}%"))
    m = j("track_b_production/bundle_manifest.json")
    out.append(("profiles", f"{m['profiles']['n_customers']:,}"))
    out.append(("keys", f"{m['n_keys']:,}"))
    out.append(("equivalence", f"{m['equivalence']['identical_order']}/{m['equivalence']['requests']} identical orderings"))
    eq = j("track_b_production/equivalence_after_optimization.json")["summary"]
    assert eq["identical_order"] == eq["requests"] == 400
    out.append(("post-opt score diff", f"{eq['max_score_abs_diff']:.1e}"))
    b0, b1 = j("track_b_production/bench_serving_baseline.json"), j("track_b_production/bench_serving_final.json")
    out.append(("baseline uncached p95", f"{b0['cold']['c1']['p95_ms']:.1f} ms"))
    out.append(("final uncached p95", f"{b1['cold']['c1']['p95_ms']:.1f} ms"))
    out.append(("final c4 rps", f"| {b1['cold']['c4']['throughput_rps']:.0f} |"))
    out.append(("baseline c4 rps", f"| {b0['cold']['c4']['throughput_rps']:.0f} |"))
    assert b1["cold"]["c16"]["errors"] + b1["cold"]["c16"]["timeouts"] == 0
    out.append(("baseline c16 errors", f"| {b0['cold']['c16']['errors'] + b0['cold']['c16']['timeouts']} |"))
    out.append(("warm p95", f"{b1['warm']['c1']['p95_ms']:.1f} ms"))
    out.append(("final RSS", f"{b1['warm']['rss_steady_mb'] / 1000:.2f} GB"))    # decimal GB, as reported
    out.append(("exact search", f"{b1['exact_vector_search']['exact_top8_ms_per_query']:.2f} ms"))
    out.append(("visual warm p95", f"{b1['warm']['visual']['warm_c1']['p95_ms']:.0f} ms"))
    # D-043 thresholds actually met
    assert b1["cold"]["c1"]["p95_ms"] <= 60 and b1["cold"]["c4"]["throughput_rps"] >= 40
    assert b1["warm"]["c1"]["p95_ms"] <= 10 and b1["cold"]["startup_to_ready_s"] <= 5
    assert b1["warm"]["rss_steady_mb"] <= 2048 and b1["warm"]["visual"]["warm_c1"]["p95_ms"] <= 100
    mix = b1["warm"]["mix"]
    items = {kk.split("=")[1]: v for kk, v in mix.items() if kk.startswith("ctl_items")}
    tot = sum(items.values())
    out.append(("prov visual", f"{items['visual_compatibility'] / tot * 100:.1f}%"))
    out.append(("prov co", f"{items['co_purchase'] / tot * 100:.1f}%"))
    return out


def main() -> int:
    text = REPORT.read_text().replace("\u2212", "-")   # typographic minus signs
    bad = [(name, s) for name, s in checks() if s not in text]
    n = len(checks())
    for name, s in bad:
        print(f"MISMATCH {name}: expected {s!r} in the report")
    print(f"{n - len(bad)}/{n} headline figures verified")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

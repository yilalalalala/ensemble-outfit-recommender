"""Assert the decision and the headline figures of reports/TRACK_A_RANKER_RETRAIN_REPORT.md against
reports/track_a_retrain/comparison.json and shap_checks.json.

  .venv/bin/python scripts/verify_ranker_retrain_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "reports" / "track_a_retrain"
REPORT = ROOT / "reports" / "TRACK_A_RANKER_RETRAIN_REPORT.md"


def pct(x, d=2):
    return f"{x * 100:+.{d}f}%"


def ci(c):
    return f"[{c[0] * 100:+.2f}, {c[1] * 100:+.2f}]"


def checks() -> list[tuple[str, str]]:
    c = json.loads((D / "comparison.json").read_text())
    s = json.loads((D / "shap_checks.json").read_text())
    v = c["variants"]
    out = []
    # The decision is the frozen rule's, applied mechanically.
    rp = c["reproduction_of_r0"]
    assert rp["ap_identical"] and rp["cand_hits_identical"] and rp["rows"] == 446056, rp
    assert c["adopted"] is None and c["passing"] == [], (c["adopted"], c["passing"])
    assert c["best_quality_only"] == "R1"
    assert c["r3_trigger"]["triggered"] is False and c["r3_trigger"]["other_gates_pass"] is False
    assert "R3" not in v
    for name in ("R1", "R2"):
        g = v[name]["gates"]
        assert not g["map_gain_ok"] and not g["map_ci_above_zero"] and g["candidate_recall_identical"]
        assert g["cost_ok"] and g["recall12_lcb_ok"] and g["ndcg12_lcb_ok"]
        assert abs(v[name]["pooled"]["candidate_recall"] - c["r0"]["pooled"]["candidate_recall"]) < 1e-12
    assert not v["R1"]["gates"]["segments_ok"] and v["R2"]["gates"]["segments_ok"]
    for name in ("R1", "R2"):
        b = v[name]["bootstrap_vs_r0"]
        out += [(f"{name} MAP", f"{v[name]['pooled']['map@12']:.5f}"),
                (f"{name} MAP rel", f"{pct(b['map@12']['relative'])} {ci(b['map@12']['relative_ci95'])}"),
                (f"{name} Recall rel", f"{pct(b['recall@12']['relative'])} {ci(b['recall@12']['relative_ci95'])}"),
                (f"{name} NDCG rel", f"{pct(b['ndcg@12']['relative'])} {ci(b['ndcg@12']['relative_ci95'])}"),
                (f"{name} new MAP", f"{pct(b['map@12_new']['relative'])} {ci(b['map@12_new']['relative_ci95'])}"),
                (f"{name} tail", f"{pct(b['recall@12_tail']['relative'])} {ci(b['recall@12_tail']['relative_ci95'])}"),
                (f"{name} folds", f"{v[name]['gates']['folds_won']}/6"),
                (f"{name} train ratio", f"{v[name]['cost']['training_time_ratio']:.2f}×"),
                (f"{name} scoring time", pct(v[name]["cost"]["scoring_time_increase"], 1)),
                (f"{name} scoring rss", pct(v[name]["cost"]["scoring_rss_increase"], 1))]
    out += [("R0 MAP", f"{c['r0']['pooled']['map@12']:.5f}"),
            ("candidate recall", f"{c['r0']['pooled']['candidate_recall']:.4f}"),
            ("repro rows", f"{rp['rows']:,}"),
            ("R1 segment lcb", f"{v['R1']['gates']['segment_lcbs']['map@12_new'] * 100:+.2f}%"),
            ("prov gain share", f"{sum(v['R2']['prov_gain_share'].values()) * 100:.2f}%")]
    sd = c["seeds"]["R1"]
    assert set(sd["per_seed_map@12"]) == {"42", "43", "44"}
    gains = sd["map_rel_gain_vs_r0"]
    out += [("seed mean", f"{sd['mean']:.5f}"), ("seed sd", f"{sd['sd']:.5f}"),
            ("seed gain mean", pct(sum(gains.values()) / len(gains)))]
    out += [(f"seed {k} gain", pct(x)) for k, x in gains.items()]
    assert max(gains.values()) < c["decision_rule"]["min_map_rel_gain"], gains
    sv = s["variants"]
    assert all(x["unsupported_chips"] == 0 and x["additivity_max_abs_err"] < 1e-9 for x in sv.values())
    assert all(x["prov_features_map_to_no_reason"] for x in sv.values())
    out += [("shap rows", f"{sv['R2']['rows']:,}"),
            ("shap prov share", f"{sv['R2']['prov_share_total'] * 100:.2f}%")]
    out += [(f"appended {k}", f"{x['top_k_share_appended'] * 100:.1f}%") for k, x in sv.items()]
    return out


def main() -> int:
    text = REPORT.read_text().replace("−", "-")
    cs = checks()
    bad = [(n, s) for n, s in cs if s.replace("−", "-") not in text]
    for n, s in bad:
        print(f"MISMATCH {n}: expected {s!r}")
    print(f"{len(cs) - len(bad)}/{len(cs)} headline figures verified")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

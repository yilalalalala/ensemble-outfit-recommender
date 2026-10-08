"""Assert the headline figures of reports/TRACK_A_RETRIEVAL_UPGRADE_REPORT.md against
reports/track_a_retrieval/comparison.json (and the adoption recorded in the config).

  .venv/bin/python scripts/verify_retrieval_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "TRACK_A_RETRIEVAL_UPGRADE_REPORT.md"


def pct(x, d=2):
    return f"{x * 100:+.{d}f}%"


def ci(c):
    return f"[{c[0] * 100:+.2f}, {c[1] * 100:+.2f}]"


def checks() -> list[tuple[str, str]]:
    r = json.loads((ROOT / "reports" / "track_a_retrieval" / "comparison.json").read_text())
    cfg = yaml.safe_load((ROOT / "configs" / "track_a_research.yaml").read_text())["research"]["retrieval_upgrade"]
    cf = r["configs"]
    a = cf["sasrec_A300"]
    b = a["bootstrap_vs_final"]
    out = []
    # The decision is the rule's, and the config records it.
    assert r["adopted"] == "sasrec_A300" == cfg["adopted"], (r["adopted"], cfg["adopted"])
    assert set(r["passing"]) == {"sasrec_A300", "F300", "neural3_A300"}, r["passing"]
    assert r["sasrec_gate"]["useful"] is True
    assert all(not cf[n]["decision_gates"]["pass"] for n in cf if n != "final" and n.endswith(("500", "1000")))
    out += [("final candidate recall", f"{cf['final']['pooled']['candidate_recall']:.4f}"),
            ("adopted candidate recall", f"{a['pooled']['candidate_recall']:.4f}"),
            ("candidate recall rel", f"{pct(b['candidate_recall']['relative'])} {ci(b['candidate_recall']['relative_ci95'])}"),
            ("MAP rel", f"{pct(b['map@12']['relative'])} {ci(b['map@12']['relative_ci95'])}"),
            ("Recall@12 rel", f"{pct(b['recall@12']['relative'])} {ci(b['recall@12']['relative_ci95'])}"),
            ("NDCG rel", pct(b["ndcg@12"]["relative"])),
            ("time ratio", f"{a['cost']['time_ratio_vs_final']:.2f}×"),
            ("rss", f"+{a['cost']['rss_increase_vs_final'] * 100:.0f}%"),
            ("tail cand", pct(b["tail_candidate_recall"]["relative"])),
            ("tail R12", f"{pct(b['tail_recall@12']['relative'])} {ci(b['tail_recall@12']['relative_ci95'])}"),
            ("conversion", f"{a['marginal_conversion'] * 100:.1f}%")]
    assert a["decision_gates"]["folds_with_candidate_gain"] == 6
    assert b["new_customer_map@12"]["relative"] == 0 and b["new_customer_candidate_recall"]["relative"] == 0
    f3 = cf["F300"]["bootstrap_vs_final"]
    out += [("F300 MAP", f"{pct(f3['map@12']['relative'])} {ci(f3['map@12']['relative_ci95'])}"),
            ("F300 new MAP", f"{f3['new_customer_map@12']['relative'] * 100:.2f}% "
                             f"[{f3['new_customer_map@12']['relative_ci95'][0] * 100:.2f}, "
                             f"{f3['new_customer_map@12']['relative_ci95'][1] * 100:.2f}]"),
            ("F300 time", f"{cf['F300']['cost']['time_ratio_vs_final']:.2f}×"),
            ("F300 conversion", f"{cf['F300']['marginal_conversion'] * 100:.1f}%")]
    assert f3["new_customer_map@12"]["relative_ci95"][1] < 0
    best = max((n for n in cf if n != "final"), key=lambda n: cf[n]["pooled"]["map@12"])
    assert best == "F300" == r["best_quality_only"]
    d = r["diagnostics"]
    mt = d["miss_taxonomy"]
    out += [("no channel", f"{mt['no_channel_proposed_eligible'] * 100:.1f}%"),
            ("cut by cap", f"{mt['proposed_then_cut_by_cap'] * 100:.1f}%"),
            ("no channel incl neural", f"{mt['no_channel_incl_neural_top1000'] * 100:.1f}%"),
            ("reconcile hits", f"{d['reconcile']['recomputed_hits']:,}")]
    assert d["reconcile"]["recomputed_hits"] == d["reconcile"]["stored_hits"]
    assert abs(d["reconcile"]["candidate_recall"] - 0.1757) < 5e-5
    rp = r["reproduction_of_final"]
    out += [("repro share", f"{rp['ap_identical_share'] * 100:.4f}%"), ("repro n", f"{rp['customers']:,}")]
    out.append(("final cost", f"{cf['final']['cost']['rank_score_seconds']:.0f} s, "
                              f"{cf['final']['cost']['peak_rss_bytes'] / 1e9:.2f} GB"))
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

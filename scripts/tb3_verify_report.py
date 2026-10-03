"""Check every headline number in the Round-3 Track B report against the stored JSON.

  .venv/bin/python scripts/tb3_verify_report.py

Exits non-zero on the first mismatch. The report is written by hand around tables
rendered by ``scripts/tb3_tables.py``; this is the guard that a hand-typed figure
has not drifted from the artifact it came from.
"""
from __future__ import annotations

import json
import pathlib
import sys

D = pathlib.Path(__file__).resolve().parents[1] / "reports" / "track_b_round3"


def load(name: str) -> dict:
    return json.loads((D / f"{name}.json").read_text())


CHECKS: list[tuple[bool, str, float, float]] = []


def chk(name: str, claimed: float, actual: float, tol: float = 5e-5) -> None:
    CHECKS.append((abs(claimed - actual) <= tol, name, claimed, actual))


def main() -> int:
    val, test = load("backtest_val_ablations"), load("backtest_test")
    sr, pl, la, tw, sv = (load("backtest_val_serving_rules"), load("protocol_leak"),
                          load("label_audit"), load("ablation_towers"), load("serving"))

    s, b = val["summary"]["systems"], val["summary"]["pooled_bootstrap_vs_shipped"]
    chk("val shipped R@12", 0.1066, s["shipped_rrf_hybrid"]["recall@12"]["mean"])
    chk("val compat R@12", 0.1379, s["lgbm_compatibility"]["recall@12"]["mean"])
    chk("val pers R@12", 0.1573, s["lgbm_personalized"]["recall@12"]["mean"])
    chk("val union R@12", 0.1088, s["rrf_all_sources_union"]["recall@12"]["mean"])
    chk("val compat lift", 0.2933, s["lgbm_compatibility"]["relative_lift_recall@12_vs_shipped"]["mean"])
    chk("val pers lift", 0.4774, s["lgbm_personalized"]["relative_lift_recall@12_vs_shipped"]["mean"])
    chk("val boot pers R@12", 0.4758, b["lgbm_personalized"]["recall@12"]["relative"])
    chk("val boot pers NDCG@12", 0.5445, b["lgbm_personalized"]["ndcg@12"]["relative"])
    chk("val boot compat R@12", 0.2937, b["lgbm_compatibility"]["recall@12"]["relative"])
    chk("union recall", 0.4285, val["summary"]["union_recall"]["mean"])
    chk("candidates per query", 189, round(val["summary"]["candidates_per_query"]["mean"]), 0.5)
    chk("val pers tail", 0.0912, s["lgbm_personalized"]["recall@12_tail"]["mean"])
    chk("val pers returning", 0.1574, s["lgbm_personalized"]["recall@12_returning"]["mean"])
    chk("val pers new customer", 0.1557, s["lgbm_personalized"]["recall@12_new_customer"]["mean"])

    base = s["lgbm_personalized"]["recall@12"]["mean"]
    for group, claimed in [("lgbm_compatibility", -0.1233), ("lgbm_minus_popularity", -0.1050),
                           ("lgbm_minus_towers", -0.0397), ("lgbm_minus_style_backoff", -0.0077),
                           ("lgbm_minus_repeat", -0.0074), ("lgbm_minus_price_colour", -0.0070),
                           ("lgbm_minus_clip", -0.0066), ("lgbm_minus_decay", -0.0062),
                           ("lgbm_minus_association", -0.0042)]:
        chk(f"ablation {group}", claimed, (s[group]["recall@12"]["mean"] - base) / base)

    t, tb = test["summary"]["systems"], test["summary"]["pooled_bootstrap_vs_shipped"]
    chk("test shipped R@12", 0.1261, t["shipped_rrf_hybrid"]["recall@12"]["mean"])
    chk("test compat R@12", 0.1596, t["lgbm_compatibility"]["recall@12"]["mean"])
    chk("test pers R@12", 0.1820, t["lgbm_personalized"]["recall@12"]["mean"])
    chk("test compat lift", 0.2657, tb["lgbm_compatibility"]["recall@12"]["relative"])
    chk("test pers lift", 0.4433, tb["lgbm_personalized"]["recall@12"]["relative"])
    chk("test pers NDCG lift", 0.4945, tb["lgbm_personalized"]["ndcg@12"]["relative"])
    chk("test union recall", 0.4719, test["per_fold"]["2020-09-16"]["union_recall"])
    chk("test queries", 86350, tb["lgbm_personalized"]["recall@12"]["n_queries"], 0.5)
    chk("test customers", 21283, tb["lgbm_personalized"]["recall@12"]["n_customers"], 0.5)
    assert test["folds"] == ["2020-09-16"], test["folds"]

    r = sr["summary"]["systems"]
    b0 = r["lgbm_compatibility"]
    for variant, claimed in [("one_colourway", -0.0868), ("max2_per_type", -0.0633),
                             ("shipped", -0.0681), ("shipped_strict", -0.3745)]:
        chk(f"serving rule {variant}", claimed,
            (r[f"lgbm_compatibility+{variant}"]["recall@12"]["mean"] - b0["recall@12"]["mean"])
            / b0["recall@12"]["mean"])
    chk("serving rule shipped diversity", 0.4910,
        (r["lgbm_compatibility+shipped"]["diversity_product_type@12"]["mean"]
         - b0["diversity_product_type@12"]["mean"]) / b0["diversity_product_type@12"]["mean"])
    chk("serving rule strict list length", 8.32,
        r["lgbm_compatibility+shipped_strict"]["mean_list_length"]["mean"], 5e-3)

    for name, claimed in [("slot_popularity", -0.0511), ("assoc_npmi", -0.0511),
                          ("shipped_rrf_hybrid", -0.0507), ("rrf_all_sources", -0.0505)]:
        chk(f"oracle catalogue {name}", claimed, pl["summary"][name]["mean_oracle_inflation"])
    chk("truth dropped as ineligible", 0.0587,
        pl["summary"]["truth_coverage"]["honest_mean_truth_dropped_share"])

    chk("label repeat article", 0.0343, la["summary"]["share_truth_bought_before"]["mean"])
    chk("label repeat style", 0.0795, la["summary"]["share_truth_style_bought_before"]["mean"])
    chk("label returning share", 0.9342, la["summary"]["share_pairs_returning_customer"]["mean"])
    chk("label truth pairs", 882947, la["summary"]["truth_pairs_total"], 0.5)

    chk("towers in_batch only", -0.2648, tw["summary"]["in_batch|mps"]["relative_vs_in_batch+logq"])
    chk("towers hard negatives", 0.0094,
        tw["summary"]["in_batch+logq+hard4|mps"]["relative_vs_in_batch+logq"])
    assert tw["determinism"]["cpu_identical"] is True
    assert tw["determinism"]["mps_identical"] is False

    chk("serving rows", 1255920, sv["n_rows"], 0.5)
    chk("serving keys", 156990, sv["n_keys"], 0.5)
    chk("serving co_purchase rows", 35794, sv["source_mix"]["co_purchase"], 0.5)
    chk("serving style_co_purchase rows", 63801, sv["source_mix"]["style_co_purchase"], 0.5)
    chk("serving visual rows", 762188, sv["source_mix"]["visual_compatibility"], 0.5)
    chk("serving popularity-fallback rows", 394137, sv["source_mix"]["popular_in_slot"], 0.5)
    chk("serving distinct types per module", 3.86,
        sv["mean_distinct_product_types_per_module"], 5e-3)
    assert "other" not in sv["source_mix"], sv["source_mix"]
    assert sv["model"] == "lgbm_compatibility" and sv["diversity_fill_back"] is True

    bad = [c for c in CHECKS if not c[0]]
    for ok, name, claimed, actual in bad:
        print(f"MISMATCH {name}: report {claimed} vs artifact {actual}")
    print(f"{len(CHECKS) - len(bad)}/{len(CHECKS)} report numbers verified against the artifacts")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

"""What Track B's label actually contains (Round 3, Phase 4.5).

  python -m ensemble.completion.label_audit [n_weeks]

The Complete-the-Look label is "another article the same customer bought in the
same slot on the same day" (DESIGN §5.2). Two things about it decide how the
personalization result should be read, and neither is visible in a recall number:

1. **How much of the label is repeat purchase.** If a large share of truth
   articles are ones the customer already bought before the cutoff, then part of
   any personalized model's advantage is "buy it again in the other slot" rather
   than outfit compatibility. The ``lgbm_minus_repeat`` ablation tests whether the
   ranker leans on it; this script says how much of the label it could lean on.

2. **How noisy the basket proxy is.** Same-day purchases are not necessarily an
   outfit: repeated quantities, several colourways of one style, and very
   heterogeneous baskets are all proxies for a basket that is a shopping trip
   rather than a look. ``ensemble.completion.association.basket_audit`` measures
   those per week; here they are tied to the labels they produce.

Descriptive only — nothing here feeds a model.
"""
from __future__ import annotations

import json
import sys
from datetime import timedelta

import numpy as np

from ensemble.completion import association as A
from ensemble.completion import protocol as P
from ensemble.config import load_config
from ensemble.data.splits import Week, load_splits
from ensemble.db import connect


def _d(x) -> str:
    return f"DATE '{x}'"


def audit_week(con, cfg, week: Week) -> dict:
    """Per-week label composition over the leakage-free eligible catalogue."""
    c = cfg.track_b
    hist_lo = week.start - timedelta(weeks=int(c.get("customer_weeks", 104)))
    con.execute(f"""CREATE OR REPLACE TEMP TABLE _la_uni AS
                    SELECT article_id FROM _la_uni_df""")
    row = con.execute(f"""
        WITH b AS ({A.basket_sql(week.start, week.end, int(c.min_basket_size),
                                 int(c.max_basket_size), week.cutoff)}),
        pairs AS (
            SELECT x.customer_idx, x.t_dat, x.article_id AS anchor, y.slot AS target_slot,
                   y.article_id AS truth, y.product_code AS truth_code
            FROM b x JOIN b y ON x.customer_idx = y.customer_idx AND x.t_dat = y.t_dat
                              AND x.slot <> y.slot
            WHERE y.article_id IN (SELECT article_id FROM _la_uni)),
        pr AS (SELECT DISTINCT customer_idx, t_dat, anchor, target_slot, truth, truth_code FROM pairs),
        hist AS (SELECT DISTINCT customer_idx, article_id FROM transactions
                 WHERE t_dat >= {_d(hist_lo)} AND t_dat < {_d(week.start)}),
        hist_code AS (SELECT DISTINCT t.customer_idx, a.product_code FROM transactions t
                      JOIN articles a USING (article_id)
                      WHERE t.t_dat >= {_d(hist_lo)} AND t.t_dat < {_d(week.start)}),
        any_hist AS (SELECT DISTINCT customer_idx FROM hist)
        SELECT count(*) AS truth_pairs,
               count(DISTINCT pr.customer_idx) AS customers,
               avg((h.article_id IS NOT NULL)::INT) AS share_truth_bought_before,
               avg((hc.product_code IS NOT NULL)::INT) AS share_truth_style_bought_before,
               avg((ah.customer_idx IS NOT NULL)::INT) AS share_pairs_returning_customer,
               avg(CASE WHEN ah.customer_idx IS NOT NULL
                        THEN (h.article_id IS NOT NULL)::INT END) AS share_truth_bought_before_returning
        FROM pr
        LEFT JOIN hist h ON h.customer_idx = pr.customer_idx AND h.article_id = pr.truth
        LEFT JOIN hist_code hc ON hc.customer_idx = pr.customer_idx AND hc.product_code = pr.truth_code
        LEFT JOIN any_hist ah ON ah.customer_idx = pr.customer_idx""").df().iloc[0].to_dict()
    return {k: (int(v) if k in ("truth_pairs", "customers") else float(v)) for k, v in row.items()}


def run(n_weeks: int = 6) -> dict:
    cfg = load_config()
    con = connect(cfg, read_only=True)
    splits = load_splits(con, cfg)
    weeks = [splits.val.shift(-k) for k in range(n_weeks - 1, -1, -1)]
    per_week = {}
    for week in weeks:
        A.mine(con, week, cfg, cfg.track_b.get("basket_filter", ""))
        uni = P.eligible_universe(con, week, cfg)
        con.register("_la_uni_df", uni[["article_id"]])
        res = audit_week(con, cfg, week)
        res["basket_noise"] = A.basket_audit(con, week, cfg)
        per_week[str(week.start)] = res
        print(f"[{week.start}] {res['truth_pairs']:,} truth pairs, "
              f"{res['share_truth_bought_before']:.1%} already bought by that customer, "
              f"{res['share_truth_style_bought_before']:.1%} same style already bought, "
              f"{res['share_pairs_returning_customer']:.1%} from returning customers", flush=True)
    keys = ["share_truth_bought_before", "share_truth_style_bought_before",
            "share_pairs_returning_customer", "share_truth_bought_before_returning"]
    summary = {k: {"mean": float(np.mean([per_week[w][k] for w in per_week])),
                   "per_week": [round(per_week[w][k], 5) for w in per_week]} for k in keys}
    summary["truth_pairs_total"] = int(sum(per_week[w]["truth_pairs"] for w in per_week))
    out = {"weeks": list(per_week), "per_week": per_week, "summary": summary,
           "manifest": P.manifest(cfg, con, {"n_weeks": n_weeks})}
    d = cfg.path("reports") / "track_b_round3"
    d.mkdir(parents=True, exist_ok=True)
    (d / "label_audit.json").write_text(json.dumps(out, indent=2, default=str))
    print("\n== label composition, mean over weeks ==")
    for k in keys:
        print(f"{k:42s} {summary[k]['mean']:.2%}")
    return out


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 6)

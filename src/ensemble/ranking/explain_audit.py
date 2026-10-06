"""Faithfulness audit of reason chips (Phase 2, DESIGN §6).

  python -m ensemble.ranking.explain_audit <model file stem> [n_customers=3000]

For a deterministic sample of validation-week buyers, ranks their candidates
with a saved validation model, explains the top 12 and compares the SHAP-only
rule (top positive reason groups) with the evidence-gated rule. Reports, per
reason, how often the SHAP-only rule would have shown a chip that the raw
features do not support, and the evidence-type mix of the gated chips.
"""
from __future__ import annotations

import json
from collections import Counter

import lightgbm as lgb

from ensemble.config import load_config
from ensemble.data.splits import load_splits
from ensemble.db import connect
from ensemble.ranking.explain import REASON_TYPE, explain, supported
from ensemble.ranking.train import score_frame, week_frame


def run(model: str, n_customers: int = 3000) -> dict:
    cfg = load_config()
    con = connect(cfg, read_only=True)
    week = load_splits(con, cfg).val
    booster = lgb.Booster(model_file=str(cfg.path("processed") / "models" / f"{model}.txt"))
    features = booster.feature_name()
    users = f"""SELECT DISTINCT customer_idx FROM transactions WHERE t_dat BETWEEN DATE '{week.start}' AND DATE '{week.end}'
                ORDER BY hash(customer_idx) LIMIT {int(n_customers)}"""
    df = week_frame(con, cfg, week, users, False, False)
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise SystemExit(f"model features not produced by the current config: {missing[:5]}")
    s = score_frame(booster, features, df)
    df["score"] = s["score"].to_numpy()
    top = (df.sort_values(["customer_idx", "score", "article_id"], ascending=[True, False, True])
             .groupby("customer_idx").head(12).reset_index(drop=True))
    shap_only = explain(booster, features, top, require_evidence=False)
    gated = explain(booster, features, top, require_evidence=True)
    rows = top[features].to_dict("records")
    shown, unsupported = Counter(), Counter()
    for e, r in zip(shap_only, rows):
        for reason in e["reasons"]:
            shown[reason["key"]] += 1
            if not supported(reason["key"], r):
                unsupported[reason["key"]] += 1
    gated_counts = Counter(r["key"] for e in gated for r in e["reasons"])
    n = len(top)
    out = {
        "model": model, "recommendations": n, "customers": int(top.customer_idx.nunique()),
        "shap_only": {"chips": sum(shown.values()), "unsupported": sum(unsupported.values()),
                      "unsupported_share": sum(unsupported.values()) / max(1, sum(shown.values())),
                      "by_reason": {k: {"shown": shown[k], "unsupported": unsupported[k],
                                        "unsupported_share": unsupported[k] / shown[k]} for k in sorted(shown)}},
        "gated": {"chips": sum(gated_counts.values()),
                  "recs_with_a_chip": sum(1 for e in gated if e["reasons"]) / n,
                  "by_reason": dict(gated_counts),
                  "by_evidence_type": dict(Counter(REASON_TYPE[k] for k in gated_counts.elements()))},
    }
    print(json.dumps(out, indent=1))
    d = cfg.path("reports") / "phase2"
    d.mkdir(exist_ok=True)
    (d / "explain_audit.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import sys
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3000)

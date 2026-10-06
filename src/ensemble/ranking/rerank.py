"""Re-ranking layer after the ranker: availability eligibility and diversity rules.

This is the third stage of the two-stage recommender (DESIGN §7.2): business
rules applied to the ranker's ordered list. It is evaluated separately so its
accuracy cost is visible.

**Availability proxy (no stock feed).** The dataset has no inventory data. The
proxy uses only what is known at the cutoff: an article whose most recent
observed sale is more than ``max_days_since_last_sale`` days before the cutoff
is treated as *likely unavailable* (sold out or discontinued). It never reads
label-week sales, so it cannot leak. It is a proxy: a slow seller with stock
looks the same as a discontinued article.

**Diversity rules.** Greedy pass in score order with caps: at most
``max_per_product_code`` colourways of one style and ``max_per_product_type``
items of one product type; skipped items are appended after the list is full
only if fewer than k items qualified. Ties break on article_id.

  python -m ensemble.ranking.rerank <scored name> [...]   # data/interim/scored/<name>.parquet
"""
from __future__ import annotations

import json
import math
from collections import Counter

import numpy as np
import pandas as pd

from ensemble.baselines import customer_age_bins, fill, popular, popular_by_age
from ensemble.evaluation.track_a import evaluate, load_segments, load_truth


def apply_rules(scored: pd.DataFrame, articles: pd.DataFrame, k: int = 12, max_days_since_last_sale: int | None = None,
                max_per_product_code: int | None = None, max_per_product_type: int | None = None) -> dict[int, list[int]]:
    """Return the top-k list per customer after eligibility and diversity rules."""
    s = scored
    if max_days_since_last_sale is not None:
        s = s[s["a_days_since_last_sale"] <= max_days_since_last_sale]
    s = s.sort_values(["customer_idx", "score", "article_id"], ascending=[True, False, True], kind="stable")
    if max_per_product_code is None and max_per_product_type is None:
        top = s.groupby("customer_idx", sort=False).head(k)
        return top.groupby("customer_idx", sort=False)["article_id"].agg(list).to_dict()
    # Keep a generous prefix per customer, then apply the caps in Python.
    s = s.groupby("customer_idx", sort=False).head(k * 6)
    code = articles["product_code"].reindex(s.article_id.to_numpy()).to_numpy()
    ptype = articles["product_type_no"].reindex(s.article_id.to_numpy()).to_numpy()
    cust = s.customer_idx.to_numpy()
    art = s.article_id.to_numpy()
    out: dict[int, list[int]] = {}
    bounds = np.flatnonzero(np.r_[True, cust[1:] != cust[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        chosen, skipped, nc, nt = [], [], Counter(), Counter()
        for i in range(a, b):
            if max_per_product_code is not None and nc[code[i]] >= max_per_product_code:
                skipped.append(art[i])
                continue
            if max_per_product_type is not None and nt[ptype[i]] >= max_per_product_type:
                skipped.append(art[i])
                continue
            chosen.append(art[i])
            nc[code[i]] += 1
            nt[ptype[i]] += 1
            if len(chosen) == k:
                break
        out[int(cust[a])] = (chosen + skipped)[:k]
    return out


def diversity(preds: dict[int, list[int]], articles: pd.DataFrame, pop_share: pd.Series) -> dict:
    """Intra-list diversity, catalogue coverage and novelty of the shown lists."""
    lists = list(preds.values())
    allrec = [a for lst in lists for a in lst]
    n_types = [len(set(articles["product_type_no"].reindex(lst))) for lst in lists]
    n_codes = [len(set(articles["product_code"].reindex(lst))) for lst in lists]
    share = pop_share.reindex(allrec).fillna(pop_share.min() if len(pop_share) else 1e-9).to_numpy()
    return {"distinct_product_types_per_list": float(np.mean(n_types)),
            "distinct_styles_per_list": float(np.mean(n_codes)),
            "catalog_coverage_articles": len(set(allrec)),
            "novelty_bits": float(np.mean(-np.log2(share))) if len(share) else math.nan}


def run(names: list[str], policies: dict | None = None) -> dict:
    from ensemble.config import load_config
    from ensemble.data.splits import Week, load_splits
    from ensemble.db import connect
    cfg = load_config()
    con = connect(cfg, read_only=True)
    splits = load_splits(con, cfg)
    articles = con.execute("SELECT article_id, product_code, product_type_no FROM articles").df().set_index("article_id")
    policies = policies or dict(cfg.rerank.policies)
    out = {}
    for name in names:
        scored = pd.read_parquet(cfg.path("interim") / "scored" / f"{name}.parquet")
        week_start = name.rsplit("_", 1)[-1]
        week = splits.val if not week_start.startswith("20") else Week(pd.Timestamp(week_start).date(),
                                                                       pd.Timestamp(week_start).date() + pd.Timedelta(days=6))
        if name.startswith("test"):
            week = splits.test
        truth, seg = load_truth(con, week), load_segments(con, week)
        pop, pop_age = popular(con, week), popular_by_age(con, week)
        ages = customer_age_bins(con, list(truth))
        ps = con.execute("""SELECT article_id, count(*) / sum(count(*)) OVER () AS s FROM transactions
                            WHERE t_dat >= ? AND t_dat < ? GROUP BY 1""",
                         [week.start - pd.Timedelta(days=7), week.start]).df().set_index("article_id")["s"]
        res = {}
        for pname, pol in policies.items():
            p = apply_rules(scored, articles, **(pol or {}))
            p = {u: fill(p.get(u, []), pop_age.get(ages.get(u, -1), pop)) for u in truth}
            m = evaluate(p, truth, seg)
            res[pname] = {"map@12": m["map@12"], "map@12_returning": m["map@12_returning"],
                          "map@12_new_customers": m["map@12_new_customers"], "recall@12": m["recall@12"],
                          **diversity(p, articles, ps)}
            print(f"{name} {pname:28s} MAP@12={m['map@12']:.5f} types/list={res[pname]['distinct_product_types_per_list']:.2f} "
                  f"styles/list={res[pname]['distinct_styles_per_list']:.2f} coverage={res[pname]['catalog_coverage_articles']} "
                  f"novelty={res[pname]['novelty_bits']:.2f}", flush=True)
        out[name] = res
    d = cfg.path("reports") / "phase2"
    d.mkdir(exist_ok=True)
    stem = names[0].rsplit("_20", 1)[0] if len(names) > 1 else names[0]
    (d / f"rerank_{stem}.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import sys
    run(sys.argv[1:])

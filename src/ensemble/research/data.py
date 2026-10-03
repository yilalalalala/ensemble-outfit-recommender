"""Point-in-time training inputs for the neural baselines (D-003).

Every function takes the label ``week`` as a required argument and reads only
``transactions.t_dat < week.start``. Outputs are plain numpy arrays so the model
process (PyTorch) never needs the database.

- ``interactions``: unique (customer, article) positives in a window, for BPR-MF and
  LightGCN, with the customer and article index spaces they define.
- ``sequences``: each customer's chronological purchases (one token per distinct
  (article, day); same-day items ordered by ``article_id`` because the data carries no
  time of day), restricted to an article vocabulary, last ``max_len`` tokens kept.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from ensemble.data.splits import Week


def _d(x) -> str:
    return f"DATE '{x}'"


def interactions(con, week: Week, window_weeks: int, min_item_count: int) -> dict[str, np.ndarray]:
    """Positives for collaborative filtering.

    Articles need ``min_item_count`` distinct buyers in the window; customers need at least
    one purchase of such an article. Returns ``users`` and ``items`` (sorted ids defining the
    index spaces) and ``u``/``i`` (index pairs, sorted by (u, i))."""
    lo = week.start - timedelta(weeks=int(window_weeks))
    z = con.execute(f"""
        WITH x AS (SELECT DISTINCT customer_idx, article_id FROM transactions
                   WHERE t_dat >= {_d(lo)} AND t_dat < {_d(week.start)}),
        items AS (SELECT article_id FROM x GROUP BY 1 HAVING count(*) >= {int(min_item_count)})
        SELECT customer_idx, article_id FROM x JOIN items USING (article_id) ORDER BY customer_idx, article_id
    """).fetchnumpy()
    users, ui = np.unique(z["customer_idx"].astype(np.int64), return_inverse=True)
    items, ii = np.unique(z["article_id"].astype(np.int64), return_inverse=True)
    return {"users": users, "items": items, "u": ui.astype(np.int32), "i": ii.astype(np.int32)}


def vocabulary(con, week: Week, vocab_weeks: int, min_item_count: int) -> np.ndarray:
    """Sorted article ids with ``min_item_count`` purchases in the ``vocab_weeks`` before the cutoff."""
    lo = week.start - timedelta(weeks=int(vocab_weeks))
    return np.sort(con.execute(f"""SELECT article_id FROM transactions
                                   WHERE t_dat >= {_d(lo)} AND t_dat < {_d(week.start)}
                                   GROUP BY 1 HAVING count(*) >= {int(min_item_count)}""").fetchnumpy()["article_id"]
                   .astype(np.int64))


def sequences(con, week: Week, vocab: np.ndarray, history_weeks: int, max_len: int) -> dict[str, np.ndarray]:
    """Last ``max_len`` vocabulary tokens per customer (left-padded with 0; token = 1 + vocab index).

    Also returns ``last_day`` (days before the cutoff of each customer's most recent token),
    used to choose training customers without reading the label week."""
    lo = week.start - timedelta(weeks=int(history_weeks))
    con.register("_vocab", pd.DataFrame({"article_id": vocab.astype(np.int64),
                                         "tok": np.arange(1, len(vocab) + 1, dtype=np.int32)}))
    z = con.execute(f"""
        WITH x AS (SELECT DISTINCT customer_idx, article_id, t_dat FROM transactions
                   WHERE t_dat >= {_d(lo)} AND t_dat < {_d(week.start)}),
        t AS (SELECT x.customer_idx, v.tok, x.t_dat, x.article_id,
                     row_number() OVER (PARTITION BY x.customer_idx ORDER BY x.t_dat DESC, x.article_id DESC) AS rn
              FROM x JOIN _vocab v USING (article_id))
        SELECT customer_idx, tok, rn, date_diff('day', t_dat, {_d(week.start)}) AS days_ago
        FROM t WHERE rn <= {int(max_len)} ORDER BY customer_idx, rn DESC
    """).fetchnumpy()
    con.unregister("_vocab")
    cust = z["customer_idx"].astype(np.int64)
    users, start = np.unique(cust, return_index=True)
    lengths = np.diff(np.r_[start, len(cust)])
    seq = np.zeros((len(users), int(max_len)), dtype=np.int32)
    # Rows arrive oldest-to-newest within a customer (rn DESC): place them right-aligned.
    pos_in_user = np.arange(len(cust)) - np.repeat(start, lengths)
    col = int(max_len) - np.repeat(lengths, lengths) + pos_in_user
    seq[np.repeat(np.arange(len(users)), lengths), col] = z["tok"].astype(np.int32)
    last_day = np.minimum.reduceat(z["days_ago"].astype(np.int32), start) if len(start) else np.array([], np.int32)
    return {"users": users, "seq": seq, "last_day": last_day}


def target_users(con, week: Week) -> np.ndarray:
    """Customers to score for ``week``: the label-week buyers (D-014). This selects *who* is
    evaluated; it never feeds a model input."""
    return np.sort(con.execute("SELECT DISTINCT customer_idx FROM transactions WHERE t_dat BETWEEN ? AND ?",
                               [week.start, week.end]).fetchnumpy()["customer_idx"].astype(np.int64))

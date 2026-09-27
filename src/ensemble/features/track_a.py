"""Track A ranker features (DESIGN §4, Stage 2).

Point-in-time correctness (D-003): every query reads the temp view ``_past``,
which is defined from the required ``week`` argument as transactions strictly
before ``week.start``. No query below references ``transactions`` directly.

Feature groups:
- customer: demographics, activity, price point, channel mix
- article: sales velocity and trend, price, age in catalogue, buyer age, metadata
- customer×article: prior purchases of the article, its product, its type, colour, group; price fit
- retrieval source: per-channel score and rank (from ``cand``), number of channels
"""
from __future__ import annotations

from datetime import timedelta

from ensemble.candidates.retrieval import CHANNELS
from ensemble.data.splits import Week

CATEGORICAL = ["product_type_no", "product_group", "graphical_appearance_no", "colour_group_code",
               "perceived_colour_value_id", "index_group_no", "garment_group_no", "club_status", "news_freq"]


def _d(x) -> str:
    return f"DATE '{x}'"


def define_past(con, week: Week) -> None:
    con.execute(f"CREATE OR REPLACE TEMP VIEW _past AS SELECT * FROM transactions WHERE t_dat < {_d(week.start)}")


def build_features(con, week: Week, with_labels: bool) -> "pandas.DataFrame":  # noqa: F821
    """Join features onto ``cand``. Labels (if requested) come from the label week only."""
    define_past(con, week)
    s = week.start
    d1, d4, d12 = (_d(s - timedelta(weeks=k)) for k in (1, 4, 12))

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _cf AS
        SELECT customer_idx,
               count(*) AS c_n_tx,
               count(*) FILTER (WHERE t_dat >= {d4}) AS c_n_tx_4w,
               count(DISTINCT t_dat) AS c_n_days,
               date_diff('day', max(t_dat), {_d(s)}) AS c_days_since_last,
               date_diff('day', min(t_dat), {_d(s)}) AS c_tenure_days,
               avg(price) AS c_mean_price,
               avg(price) FILTER (WHERE t_dat >= {d12}) AS c_mean_price_12w,
               avg((sales_channel_id = 2)::INT) AS c_online_share
        FROM _past WHERE customer_idx IN (SELECT DISTINCT customer_idx FROM cand) GROUP BY 1
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _af AS
        WITH ca AS (SELECT DISTINCT article_id FROM cand),
        s AS (
            SELECT p.article_id,
                   count(*) FILTER (WHERE t_dat >= {d1}) AS a_n_1w,
                   count(*) FILTER (WHERE t_dat >= {d4}) AS a_n_4w,
                   count(*) FILTER (WHERE t_dat >= {d12}) AS a_n_12w,
                   count(*) AS a_n_all,
                   date_diff('day', min(t_dat), {_d(s)}) AS a_days_since_first_sale,
                   date_diff('day', max(t_dat), {_d(s)}) AS a_days_since_last_sale,
                   avg(price) FILTER (WHERE t_dat >= {d4}) AS a_mean_price_4w,
                   avg(price) AS a_mean_price,
                   avg((sales_channel_id = 2)::INT) FILTER (WHERE t_dat >= {d4}) AS a_online_share
            FROM _past p JOIN ca USING (article_id) GROUP BY 1),
        ag AS (
            SELECT p.article_id, avg(c.age) AS a_buyer_age, stddev(c.age) AS a_buyer_age_std
            FROM _past p JOIN ca USING (article_id) JOIN customers c USING (customer_idx)
            WHERE p.t_dat >= {d4} GROUP BY 1)
        SELECT a.article_id, a.product_code, a.product_type_no,
               hash(a.product_group_name) % 1000 AS product_group,
               a.graphical_appearance_no, a.colour_group_code, a.perceived_colour_value_id,
               a.department_no, a.index_group_no, a.garment_group_no, a.section_no,
               s.* EXCLUDE (article_id),
               s.a_n_1w / (s.a_n_4w / 4.0 + 1) AS a_trend_1w_4w,
               ag.a_buyer_age, ag.a_buyer_age_std
        FROM ca JOIN articles a USING (article_id)
        LEFT JOIN s USING (article_id) LEFT JOIN ag USING (article_id)
    """)
    # Customer×article affinities from the customer's full history before the cutoff.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _ch AS
        SELECT p.customer_idx, p.article_id, p.t_dat, a.product_code, a.product_type_no,
               a.colour_group_code, a.index_group_no, a.garment_group_no
        FROM _past p JOIN articles a USING (article_id)
        WHERE p.customer_idx IN (SELECT DISTINCT customer_idx FROM cand)
    """)
    affinity = []
    for col, name in [("product_code", "prod"), ("product_type_no", "type"), ("colour_group_code", "colour"),
                      ("index_group_no", "igroup"), ("garment_group_no", "ggroup")]:
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _aff_{name} AS
            SELECT customer_idx, {col}, count(*) AS n FROM _ch GROUP BY 1, 2
        """)
        affinity.append((col, name))
    aff_select = ",\n".join(f"coalesce(aff_{n}.n, 0) AS ca_n_{n}, coalesce(aff_{n}.n, 0) / cf.c_n_tx AS ca_share_{n}"
                            for _, n in affinity)
    aff_join = "\n".join(f"LEFT JOIN _aff_{n} aff_{n} ON aff_{n}.customer_idx = c.customer_idx AND aff_{n}.{col} = af.{col}"
                         for col, n in affinity)
    channel_cols = ", ".join(f"c.{ch}_score, c.{ch}_rank" for ch in CHANNELS)
    n_channels = " + ".join(f"({ch}_rank IS NOT NULL)::INT" for ch in CHANNELS)
    label = (f"""(c.customer_idx, c.article_id) IN (
                   SELECT customer_idx, article_id FROM transactions
                   WHERE t_dat BETWEEN {_d(week.start)} AND {_d(week.end)})::INT AS label,"""
             if with_labels else "")
    df = con.execute(f"""
        WITH art AS (
            SELECT customer_idx, article_id, count(*) AS ca_n_article,
                   date_diff('day', max(t_dat), {_d(s)}) AS ca_days_since_article
            FROM _ch GROUP BY 1, 2)
        SELECT c.customer_idx, c.article_id, {label}
               {channel_cols}, {n_channels} AS n_channels,
               cu.age, hash(cu.club_member_status) % 100 AS club_status,
               hash(cu.fashion_news_frequency) % 100 AS news_freq, cu.FN, cu.Active,
               cf.* EXCLUDE (customer_idx),
               af.* EXCLUDE (article_id, product_code),
               coalesce(art.ca_n_article, 0) AS ca_n_article, art.ca_days_since_article,
               {aff_select},
               af.a_mean_price_4w / cf.c_mean_price AS ca_price_ratio,
               cu.age - af.a_buyer_age AS ca_age_gap
        FROM cand c
        JOIN customers cu ON cu.customer_idx = c.customer_idx
        LEFT JOIN _cf cf ON cf.customer_idx = c.customer_idx
        JOIN _af af ON af.article_id = c.article_id
        LEFT JOIN art ON art.customer_idx = c.customer_idx AND art.article_id = c.article_id
        {aff_join}
        ORDER BY c.customer_idx
    """).df()
    for t in ["_cf", "_af", "_ch", *(f"_aff_{n}" for _, n in affinity)]:
        con.execute(f"DROP TABLE IF EXISTS {t}")
    return compact(df)


def compact(df):
    """Ids as int32, label as int8, everything else float32 (NaN = missing)."""
    for col in df.columns:
        if col in ("customer_idx", "article_id"):
            df[col] = df[col].astype("int32")
        elif col == "label":
            df[col] = df[col].astype("int8")
        else:
            df[col] = df[col].astype("float32")
    return df

"""Track A ranker features (DESIGN §4, Stage 2).

Point-in-time correctness (D-003): every query reads the temp view ``_past``,
which is defined from the required ``week`` argument as transactions strictly
before ``week.start``. No query below references ``transactions`` directly.

**Lifetime vs recent windows.** ``_past`` holds the customer's *entire* history
before the cutoff, not the 16-week sampling window of DATA.md. Features
computed over all of it carry the suffix ``_life``; windowed features carry
their window (``_1w``, ``_4w``, ``_12w``, …); ``ca_w_*`` are recency-weighted
sums (weight 1 / (1 + days_ago / 7)) over the whole history. The difference is
deliberate and ablated (reports/PHASE2_UPGRADE_REPORT.md).

Feature groups (``features.groups`` in the config switches the optional ones):
- customer: demographics, activity, price point, channel mix
- article: sales velocity and trend, price, age in catalogue, buyer age, metadata
- customer×article: prior purchases of the article, its product, type, colour,
  index group, garment group; price fit
- ``recency_affinity`` (optional): recency-weighted customer×article affinities
  and days since the customer last bought the article's type / department
- ``short_velocity`` (optional): 1- and 3-day sales, a price-change signal
- retrieval source: per-channel score and rank (from ``cand``), number of channels

String categoricals use stable vocabularies (``ensemble.features.vocab``).
"""
from __future__ import annotations

from datetime import timedelta

from ensemble.candidates.retrieval import channel_columns
from ensemble.data.splits import Week
from ensemble.features import vocab as V

CATEGORICAL = ["product_type_no", "product_group", "graphical_appearance_no", "colour_group_code",
               "perceived_colour_value_id", "index_group_no", "garment_group_no", "club_status", "news_freq"]
DEFAULT_GROUPS = ("recency_affinity", "short_velocity")
# Customer×article affinity keys: (article column, feature suffix)
AFFINITY = [("product_code", "prod"), ("product_type_no", "type"), ("colour_group_code", "colour"),
            ("index_group_no", "igroup"), ("garment_group_no", "ggroup")]


def _d(x) -> str:
    return f"DATE '{x}'"


def define_past(con, week: Week) -> None:
    con.execute(f"CREATE OR REPLACE TEMP VIEW _past AS SELECT * FROM transactions WHERE t_dat < {_d(week.start)}")


def build_features(con, week: Week, with_labels: bool, groups=DEFAULT_GROUPS, cfg=None) -> "pandas.DataFrame":  # noqa: F821
    """Join features onto ``cand``. Labels (if requested) come from the label week only."""
    groups = set(groups)
    define_past(con, week)
    V.register(con, V.load_vocab(con, cfg))
    s = week.start
    d1, d4, d12 = (_d(s - timedelta(weeks=k)) for k in (1, 4, 12))
    d1d, d3d = (_d(s - timedelta(days=k)) for k in (1, 3))
    rw = f"1.0 / (1 + date_diff('day', t_dat, {_d(s)}) / 7.0)"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _cf AS
        SELECT customer_idx,
               count(*) AS c_n_tx_life,
               count(*) FILTER (WHERE t_dat >= {d1}) AS c_n_tx_1w,
               count(*) FILTER (WHERE t_dat >= {d4}) AS c_n_tx_4w,
               count(*) FILTER (WHERE t_dat >= {d12}) AS c_n_tx_12w,
               count(DISTINCT t_dat) AS c_n_days_life,
               date_diff('day', max(t_dat), {_d(s)}) AS c_days_since_last,
               date_diff('day', min(t_dat), {_d(s)}) AS c_tenure_days,
               avg(price) AS c_mean_price_life,
               avg(price) FILTER (WHERE t_dat >= {d12}) AS c_mean_price_12w,
               avg((sales_channel_id = 2)::INT) AS c_online_share_life
        FROM _past WHERE customer_idx IN (SELECT DISTINCT customer_idx FROM cand) GROUP BY 1
    """)
    short = (f""",
                   count(*) FILTER (WHERE t_dat >= {d1d}) AS a_n_1d,
                   count(*) FILTER (WHERE t_dat >= {d3d}) AS a_n_3d,
                   avg(price) FILTER (WHERE t_dat >= {d1}) AS a_mean_price_1w"""
             if "short_velocity" in groups else "")
    short_derived = (""",
               s.a_n_3d / (s.a_n_1w / 7.0 * 3 + 1) AS a_trend_3d_1w,
               s.a_mean_price_1w / s.a_mean_price_4w AS a_price_change_1w_4w"""
                     if "short_velocity" in groups else "")
    pg_join, pg_expr = V.encode("product_group", "a", "product_group_name")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _af AS
        WITH ca AS (SELECT DISTINCT article_id FROM cand),
        s AS (
            SELECT p.article_id,
                   count(*) FILTER (WHERE t_dat >= {d1}) AS a_n_1w,
                   count(*) FILTER (WHERE t_dat >= {d4}) AS a_n_4w,
                   count(*) FILTER (WHERE t_dat >= {d12}) AS a_n_12w,
                   count(*) AS a_n_life,
                   date_diff('day', min(t_dat), {_d(s)}) AS a_days_since_first_sale,
                   date_diff('day', max(t_dat), {_d(s)}) AS a_days_since_last_sale,
                   avg(price) FILTER (WHERE t_dat >= {d4}) AS a_mean_price_4w,
                   avg(price) AS a_mean_price_life,
                   avg((sales_channel_id = 2)::INT) FILTER (WHERE t_dat >= {d4}) AS a_online_share_4w{short}
            FROM _past p JOIN ca USING (article_id) GROUP BY 1),
        ag AS (
            SELECT p.article_id, avg(c.age) AS a_buyer_age_4w, stddev(c.age) AS a_buyer_age_std_4w
            FROM _past p JOIN ca USING (article_id) JOIN customers c USING (customer_idx)
            WHERE p.t_dat >= {d4} GROUP BY 1)
        SELECT a.article_id, a.product_code, a.department_no AS _dept, a.product_type_no,
               {pg_expr} AS product_group,
               a.graphical_appearance_no, a.colour_group_code, a.perceived_colour_value_id,
               a.department_no, a.index_group_no, a.garment_group_no, a.section_no,
               s.* EXCLUDE (article_id),
               s.a_n_1w / (s.a_n_4w / 4.0 + 1) AS a_trend_1w_4w{short_derived},
               ag.a_buyer_age_4w, ag.a_buyer_age_std_4w
        FROM ca JOIN articles a USING (article_id) {pg_join}
        LEFT JOIN s USING (article_id) LEFT JOIN ag USING (article_id)
    """)
    # Customer×article affinities from the customer's full history before the cutoff.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _ch AS
        SELECT p.customer_idx, p.article_id, p.t_dat, {rw} AS w, a.product_code, a.product_type_no,
               a.colour_group_code, a.index_group_no, a.garment_group_no, a.department_no
        FROM _past p JOIN articles a USING (article_id)
        WHERE p.customer_idx IN (SELECT DISTINCT customer_idx FROM cand)
    """)
    rec = "recency_affinity" in groups
    for col, name in AFFINITY:
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _aff_{name} AS
            SELECT customer_idx, {col}, count(*) AS n, sum(w) AS w,
                   date_diff('day', max(t_dat), {_d(s)}) AS days
            FROM _ch GROUP BY 1, 2
        """)
    if rec:
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _aff_dept AS
            SELECT customer_idx, department_no, sum(w) AS w, date_diff('day', max(t_dat), {_d(s)}) AS days
            FROM _ch GROUP BY 1, 2
        """)
        con.execute("CREATE OR REPLACE TEMP TABLE _cw AS SELECT customer_idx, sum(w) AS w FROM _ch GROUP BY 1")
    aff_select = ",\n".join(
        f"coalesce(aff_{n}.n, 0) AS ca_n_{n}_life, coalesce(aff_{n}.n, 0) / cf.c_n_tx_life AS ca_share_{n}_life"
        for _, n in AFFINITY)
    if rec:
        aff_select += ",\n" + ",\n".join(
            f"coalesce(aff_{n}.w, 0) AS ca_w_{n}, coalesce(aff_{n}.w, 0) / cw.w AS ca_wshare_{n}" for _, n in AFFINITY)
        aff_select += """,
               aff_type.days AS ca_days_since_type, aff_dept.days AS ca_days_since_dept,
               coalesce(aff_dept.w, 0) / cw.w AS ca_wshare_dept, coalesce(art.ca_w_article, 0) AS ca_w_article"""
    aff_join = "\n".join(f"LEFT JOIN _aff_{n} aff_{n} ON aff_{n}.customer_idx = c.customer_idx AND aff_{n}.{col} = af.{col}"
                         for col, n in AFFINITY)
    if rec:
        aff_join += """
        LEFT JOIN _aff_dept aff_dept ON aff_dept.customer_idx = c.customer_idx AND aff_dept.department_no = af._dept
        LEFT JOIN _cw cw ON cw.customer_idx = c.customer_idx"""
    channels = channel_columns(con)
    channel_cols = ", ".join(f"c.{ch}_score, c.{ch}_rank" for ch in channels)
    n_channels = " + ".join(f"({ch}_rank IS NOT NULL)::INT" for ch in channels)
    label = (f"""(c.customer_idx, c.article_id) IN (
                   SELECT customer_idx, article_id FROM transactions
                   WHERE t_dat BETWEEN {_d(week.start)} AND {_d(week.end)})::INT AS label,"""
             if with_labels else "")
    cs_join, cs_expr = V.encode("club_status", "cu", "club_member_status")
    nf_join, nf_expr = V.encode("news_freq", "cu", "fashion_news_frequency")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _feat AS
        WITH art AS (
            SELECT customer_idx, article_id, count(*) AS ca_n_article_life, sum(w) AS ca_w_article,
                   date_diff('day', max(t_dat), {_d(s)}) AS ca_days_since_article
            FROM _ch GROUP BY 1, 2)
        SELECT c.customer_idx, c.article_id, {label}
               {channel_cols}, {n_channels} AS n_channels,
               cu.age, {cs_expr} AS club_status, {nf_expr} AS news_freq, cu.FN, cu.Active,
               cf.* EXCLUDE (customer_idx),
               af.* EXCLUDE (article_id, product_code, _dept),
               coalesce(art.ca_n_article_life, 0) AS ca_n_article_life, art.ca_days_since_article,
               {aff_select},
               af.a_mean_price_4w / cf.c_mean_price_life AS ca_price_ratio,
               cu.age - af.a_buyer_age_4w AS ca_age_gap
        FROM cand c
        JOIN customers cu ON cu.customer_idx = c.customer_idx {cs_join} {nf_join}
        LEFT JOIN _cf cf ON cf.customer_idx = c.customer_idx
        JOIN _af af ON af.article_id = c.article_id
        LEFT JOIN art ON art.customer_idx = c.customer_idx AND art.article_id = c.article_id
        {aff_join}
    """)
    keep = {"customer_idx", "article_id", "label"}
    cols = [r[0] for r in con.execute("DESCRIBE _feat").fetchall()]
    casts = ", ".join(
        ["customer_idx::INTEGER AS customer_idx", "article_id::INTEGER AS article_id"]
        + (["label::TINYINT AS label"] if with_labels else [])
        + [f'"{c}"::FLOAT AS "{c}"' for c in cols if c not in keep])
    # Deterministic row order: customer, then article (ties in the ranker break on this order).
    df = con.execute(f"SELECT {casts} FROM _feat ORDER BY customer_idx, article_id").df()
    for t in ["_feat", "_cf", "_af", "_ch", "_cw", "_aff_dept", *(f"_aff_{n}" for _, n in AFFINITY)]:
        con.execute(f"DROP TABLE IF EXISTS {t}")
    return df


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

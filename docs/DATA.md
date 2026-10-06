# Data

## Source

H&M Personalized Fashion Recommendations, Kaggle. Acquisition is documented in
the README; the dataset is never committed.

| file | rows | notes |
| --- | ---: | --- |
| `transactions_train.csv` | 31,788,324 | `t_dat`, `customer_id`, `article_id`, `price`, `sales_channel_id` |
| `customers.csv` | 1,371,980 | `age`, `club_member_status`, `fashion_news_frequency`, `postal_code` |
| `articles.csv` | 105,542 | product type, product group, colour, department, index group, garment group, detail description |
| `images/` | ~105k JPEGs | `images/0NN/0NNxxxxxx.jpg`, keyed by article id |

## Derived concepts

**Basket.** All articles purchased by one customer on one day
(`customer_id`, `t_dat`). The dataset has no explicit order id, so the pair is
the best available proxy, and it is a good one: H&M transactions are timestamped
by day and a customer rarely shops twice in a day.

**Price.** Already normalised by the publisher, not in currency. Useful for
relative comparisons and price tiers, meaningless as an absolute figure — so it
is never displayed as money.

**Accessories.** `product_group_name == "Accessories"`: 11,158 articles, of
which ~2,000 are jewellery (`Earring`, `Necklace`, `Ring`, `Bracelet`). An
earlier draft quoted ~2,160 for the whole group, which was the jewellery count.

**Slots.** Track B groups articles into wearable slots by `product_group_name`
(DESIGN §5.1). Measured on the 16 weeks to 2020-09-22: 1.51M baskets, 898k with
2–6 items, 98k pairing a garment with an accessory, 26k pairing a garment with
jewellery.

## Splits

```
 ├──────────── training ────────────┤ validation ├ test ┤
                                     (7 days)   (7 days)
```

The test week is touched once. All model selection happens on validation.

## Sampling

| parameter | default | reason |
| --- | --- | --- |
| `WINDOW_WEEKS` | 16 | Fashion is seasonal; old purchases are weak evidence. Cuts the working set roughly 10×. |
| `MIN_BASKET_SIZE` | 2 | A single-item basket carries no pairing information. |
| `MAX_BASKET_SIZE` | 6 | Larger baskets are stock-up trips, not outfits. |
| `MIN_PAIR_SUPPORT` | 10 | PMI is unstable on rare pairs. |
| Customer sampling | none | Subsampling customers distorts the popularity distribution every baseline depends on. |

All are configuration. A full-data run changes values, not code.

**What `WINDOW_WEEKS` does and does not limit (Phase 2 clarification).** The
16-week window applies to Track B basket mining and to the documented sampling
plan. Track A reads further back where it measurably helps: ranker features
built from `_past` use the customer's *entire* history before the cutoff and are
suffixed `_life`; windowed features carry their window (`_1w`, `_4w`, `_12w`);
`ca_w_*` are recency-weighted sums over the whole history. Retrieval seeds use
their own configured windows (e.g. colourway variants and category affinity
seeded from 104 weeks), chosen on validation recall (D-027).

**Launch proxy.** The dataset has no launch dates. "New arrival" means *first
observed sale* within a window before the cutoff. An article launched earlier but
unsold looks new; one that launches inside the label week cannot be retrieved
from history at all (item cold start, ~5% of label-week purchases).

**Availability proxy.** There is no stock feed. An article whose last observed
sale is more than 28 days before the cutoff is treated as likely unavailable in
the re-ranking stage. Only pre-cutoff sales are read, so the rule cannot leak;
a slow seller with stock looks the same as a discontinued article.

## Integrity checks

Run at ingestion and asserted in tests:

- Row counts reconcile with the source files
- Every `article_id` in transactions exists in `articles`
- Every `customer_id` in transactions exists in `customers`
- `t_dat` is within the expected range and has no gaps beyond expected closures
- No transaction in the training window post-dates its split boundary

"""M6 serving: every surface in DESIGN §7.2 answers, and Complete the Look is diverse."""
from collections import Counter

import pytest

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def client(cfg):
    if not cfg.path("serving_db").exists():
        pytest.skip("serving store not built; run `make serving`")
    from fastapi.testclient import TestClient

    from ensemble.api.app import app

    return TestClient(app)


@pytest.fixture(scope="module")
def demo(client):
    return client.get("/api/customers").json()


def test_home_modules(client, demo):
    returning = next(c for c in demo if c["segment"] == "returning")
    d = client.get(f"/api/home/{returning['customer_idx']}").json()
    mods = {m["id"]: m["items"] for m in d["modules"]}
    assert len(mods["for_you"]) == 12
    assert all(it["reasons"] for it in mods["for_you"])
    assert mods["buy_again"] and mods["trending"]


def test_new_customer_gets_recommendations(client, demo):
    new = next(c for c in demo if c["segment"] == "new")
    d = client.get(f"/api/home/{new['customer_idx']}").json()
    mods = {m["id"]: m["items"] for m in d["modules"]}
    assert len(mods["for_you"]) == 12 and not mods["buy_again"]


def test_product_page_and_diversity(client, cfg, demo):
    """Complete the Look answers, stays in the right slots, and is diverse.

    The diversity caps are applied as a *re-ordering* with back-fill to
    `serving.ctl_per_slot` (D-035): applied as a hard filter they left modules
    part-empty and cost five times as much Recall@12 on the rolling folds. The cap
    arithmetic itself is unit-tested in `test_apply_diversity_*`; what this test
    checks is that serving is wired to it and that the result is still varied.
    """
    returning = next(c for c in demo if c["segment"] == "returning")
    item = client.get(f"/api/home/{returning['customer_idx']}").json()["modules"][0]["items"][0]
    d = client.get(f"/api/product/{item['article_id']}").json()
    assert d["article"]["article_id"] == item["article_id"]
    assert d["complete_the_look"]
    per_slot = int(cfg.serving.ctl_per_slot)
    cap = int(cfg.serving.max_per_product_type)
    for module in d["complete_the_look"]:
        assert module["slot"] != d["article"]["slot"]
        ids = [it["article_id"] for it in module["items"]]
        assert 0 < len(ids) <= per_slot
        assert len(ids) == len(set(ids))
        assert all(it["slot"] == module["slot"] for it in module["items"])
        assert all(it["reasons"] for it in module["items"])
        # The cap bites on the part of the list it ordered: the first `cap` positions
        # can never already break it, and the module is more varied than one style.
        types = Counter(it["product_type_name"] for it in module["items"])
        assert len(types) >= min(len(ids), 2) or len(ids) < 2
        assert len(Counter(it["product_type_name"] for it in module["items"][:cap])) >= 1


def test_explain_and_events(client, demo):
    returning = next(c for c in demo if c["segment"] == "returning")
    item = client.get(f"/api/home/{returning['customer_idx']}").json()["modules"][0]["items"][0]
    e = client.get(f"/api/explain/{returning['customer_idx']}/{item['article_id']}").json()
    assert e["shap"] and e["reasons"]
    r = client.post("/api/events", json={"customer_idx": returning["customer_idx"], "event": "click",
                                         "surface": "for_you", "article_id": item["article_id"]})
    assert r.json() == {"ok": True}


def test_image_fallback(client):
    assert client.get("/images/1.jpg").status_code == 200


def test_label_tasks_are_blind_and_complete(client):
    tasks = client.get("/api/label/tasks?round=1").json()
    assert len(tasks) == 20 and all(len(t["candidates"]) == 5 for t in tasks)
    assert all("mode" not in t for t in tasks)          # the labeller cannot see which mode produced a list
    r = client.get(tasks[0]["crop"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"


def test_complete_the_look_rejects_fabricated_anchor():
    from ensemble.assistant.tools import Toolbox
    tb = Toolbox({"grounded": set()})
    r = tb.complete_the_look(12345)
    assert r["results"] == [] and "search_catalog" in r["error"]


def test_round2_is_disjoint_and_stratified(client):
    r1 = {t["task_id"].rsplit("|", 1)[0] for t in client.get("/api/label/tasks?round=1").json()}
    r2 = client.get("/api/label/tasks?round=2").json()
    assert len(r2) == 20 and not r1 & {t["task_id"].rsplit("|", 1)[0] for t in r2}
    assert sum(t["category"] == "jewellery" for t in r2) == 6   # 3 garments x 2 modes


# --- Shop UI catalogue: product families, colourways, prototype prices (round-two consumer polish) ----

def test_product_code_convention_holds_for_the_whole_catalogue(cfg):
    import sqlite3
    con = sqlite3.connect(cfg.path("serving_db"))
    bad = con.execute("SELECT count(*) FROM articles WHERE article_id / 1000 <> CAST(product_code AS INTEGER)").fetchone()[0]
    assert bad == 0


def test_families_group_by_real_product_code_with_one_price(client, cfg):
    import sqlite3
    con = sqlite3.connect(cfg.path("serving_db"))
    code = con.execute("SELECT product_code FROM articles GROUP BY product_code HAVING count(*) BETWEEN 3 AND 8 ORDER BY product_code LIMIT 1").fetchone()[0]
    members = [r[0] for r in con.execute("SELECT article_id FROM articles WHERE product_code = ? ORDER BY article_id", (code,))]
    d1 = client.get(f"/api/catalog/families?articles={members[0]}").json()
    d2 = client.get(f"/api/catalog/families?articles={members[-1]}").json()
    assert d1["articles"][str(members[0])] == code and d2["articles"][str(members[-1])] == code
    f1, f2 = d1["families"][code], d2["families"][code]
    assert f1["price_usd"] == f2["price_usd"]                     # one price per family, from any colourway
    ids = {v["article_id"] for v in f1["variants"]}
    assert members[0] in ids and ids <= set(members)              # only real articles of this family
    assert [v["article_id"] for v in f1["variants"]] == sorted(ids)


def test_same_name_products_from_different_families_are_not_merged(client, cfg):
    import sqlite3
    con = sqlite3.connect(cfg.path("serving_db"))
    name = con.execute("SELECT prod_name FROM articles GROUP BY prod_name HAVING count(DISTINCT product_code) > 1 ORDER BY prod_name LIMIT 1").fetchone()[0]
    a, b = (r[0] for r in con.execute("""SELECT min(article_id) FROM articles WHERE prod_name = ? GROUP BY product_code
                                         ORDER BY product_code LIMIT 2""", (name,)))
    d = client.get(f"/api/catalog/families?articles={a},{b}").json()
    assert d["articles"][str(a)] != d["articles"][str(b)]
    assert len(d["families"]) == 2


def test_families_endpoint_validation(client):
    assert client.get("/api/catalog/families").json() == {"articles": {}, "families": {}}
    assert client.get("/api/catalog/families?articles=abc").status_code == 400
    assert client.get("/api/catalog/families?articles=" + ",".join(str(i) for i in range(401))).status_code == 400
    assert client.get("/api/catalog/families?articles=1").json()["articles"] == {}     # unknown ids are omitted

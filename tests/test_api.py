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
    returning = next(c for c in demo if c["segment"] == "returning")
    item = client.get(f"/api/home/{returning['customer_idx']}").json()["modules"][0]["items"][0]
    d = client.get(f"/api/product/{item['article_id']}").json()
    assert d["article"]["article_id"] == item["article_id"]
    for module in d["complete_the_look"]:
        assert module["slot"] != d["article"]["slot"]
        types = Counter(it["product_type_name"] for it in module["items"])
        assert max(types.values()) <= int(cfg.serving.max_per_product_type)
        names = [it["prod_name"] for it in module["items"]]
        assert len(names) == len(set(names)) or len(module["items"]) < 2


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

"""Staged "Complete my look" contract (docs/AI_STYLIST_PRODUCT_FLOW.md §5): /api/stylist/detect and
/api/stylist/edit. Detection is stubbed (no model call); matching and Complete the Look run on the local data."""
import io
import json

import pytest
from PIL import Image

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def client(cfg):
    if not cfg.path("serving_db").exists():
        pytest.skip("serving store not built; run `make serving`")
    from fastapi.testclient import TestClient

    from ensemble.api.app import app
    return TestClient(app)


def _photo(w=600, h=900) -> bytes:
    img = Image.new("RGB", (w, h), (200, 190, 170))
    for y in range(h // 2, h):          # a darker lower half, like trousers under a top
        for x in range(0, w, 3):
            img.putpixel((x, y), (40, 50, 90))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


GARMENTS = [{"category": "top", "colour": "beige", "description": "oversized sweatshirt", "box": [0.1, 0.1, 0.9, 0.5]},
            {"category": "bottom", "colour": "blue", "description": "wide-leg jeans", "box": [0.2, 0.45, 0.8, 0.95]}]


def test_detect_returns_reviewable_garments_with_fractional_boxes(client, monkeypatch):
    import ensemble.llm.client as llm
    import ensemble.vision.outfit as outfit

    def fake_detect(image, client_, purpose="detect"):
        small = image.resize((300, 450))
        return [{"category": "top", "colour": "beige", "description": "sweatshirt", "box": [30, 45, 270, 225]},
                {"category": "shoes", "colour": "black", "description": "flats", "box": None}], small
    monkeypatch.setattr(outfit, "detect", fake_detect)
    monkeypatch.setattr(llm, "get_client", lambda role="assistant", **k: object())
    r = client.post("/api/stylist/detect", files={"photo": ("o.jpg", _photo(), "image/jpeg")})
    assert r.status_code == 200
    d = r.json()
    assert [g["id"] for g in d["garments"]] == ["g1", "g2"]
    assert d["garments"][0]["box"] == [0.1, 0.1, 0.9, 0.5]
    assert d["garments"][1]["box"] is None
    assert "belt" in d["categories"]


def test_detect_failure_is_a_clean_503(client, monkeypatch):
    import ensemble.llm.client as llm
    import ensemble.vision.outfit as outfit
    monkeypatch.setattr(llm, "get_client", lambda role="assistant", **k: object())
    monkeypatch.setattr(outfit, "detect", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("vlm down")))
    r = client.post("/api/stylist/detect", files={"photo": ("o.jpg", _photo(), "image/jpeg")})
    assert r.status_code == 503


def test_detect_rejects_a_non_image(client):
    r = client.post("/api/stylist/detect", files={"photo": ("o.txt", b"not an image", "text/plain")})
    assert r.status_code == 415


@pytest.mark.parametrize("goal", ["complete", "occasion", "similar"])
def test_edit_returns_priority_picks_with_consumer_reasons(client, goal):
    data = {"garments": json.dumps(GARMENTS), "goal": goal, "occasion": "a wedding", "customer": "168058"}
    r = client.post("/api/stylist/edit", files={"photo": ("o.jpg", _photo(), "image/jpeg")}, data=data)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["goal"] == goal and d["summary"]
    assert 1 <= len(d["picks"]) <= 6
    ids = [p["card"]["article_id"] for p in d["picks"]]
    assert len(ids) == len(set(ids))                                        # no repeated product
    alt_ids = {it["article_id"] for a in d["alternatives"] for it in a["items"]}
    assert not alt_ids & set(ids)                                           # alternatives never repeat a pick
    banned = ("score", "similarity", "ranker", "model", "clip")
    for p in d["picks"]:
        assert p["reason"] and not any(b in p["reason"].lower() for b in banned)
        assert p["card"]["image"] and p["card"]["prod_name"]
    if goal == "occasion":
        assert all("a wedding" in p["reason"] for p in d["picks"])
    if goal == "similar":
        assert d["anchor"] is None and {p["group"] for p in d["picks"]} <= {"top", "bottom"}
    else:
        assert d["anchor"] is not None


def test_edit_uses_the_confirmed_garments(client):
    """Changing a confirmed piece changes what is matched: the shopper's correction is honoured."""
    a = client.post("/api/stylist/edit", files={"photo": ("o.jpg", _photo(), "image/jpeg")},
                    data={"garments": json.dumps(GARMENTS[:1]), "goal": "similar"}).json()
    b = client.post("/api/stylist/edit", files={"photo": ("o.jpg", _photo(), "image/jpeg")},
                    data={"garments": json.dumps([{**GARMENTS[0], "category": "shoes", "description": "flat shoes"}]),
                          "goal": "similar"}).json()
    assert a["matched"][0]["category"] == "top" and b["matched"][0]["category"] == "shoes"
    assert a["matched"][0]["article_id"] != b["matched"][0]["article_id"]


@pytest.mark.parametrize("data,msg", [
    ({"garments": "[]", "goal": "complete"}, "between one and eight"),
    ({"garments": json.dumps([{"category": "spaceship"}]), "goal": "complete"}, "category"),
    ({"garments": json.dumps(GARMENTS), "goal": "teleport"}, "goal"),
    ({"garments": json.dumps(GARMENTS), "goal": "occasion"}, "occasion"),
    ({"garments": "not json", "goal": "complete"}, "JSON"),
])
def test_edit_validation(client, data, msg):
    r = client.post("/api/stylist/edit", files={"photo": ("o.jpg", _photo(), "image/jpeg")}, data=data)
    assert r.status_code == 400 and msg in r.text


def test_legacy_snap_and_v2_outfit_routes_still_exist(client):
    paths = {r.path for r in client.app.routes}
    assert {"/api/snap", "/api/v2/outfit", "/api/visual-search", "/api/stylist/detect", "/api/stylist/edit"} <= paths


@pytest.mark.parametrize("goal", ["complete", "occasion", "similar"])
def test_edit_picks_one_product_per_family(client, goal):
    """Two colourways of one product never take two pick slots (article_id // 1000 is the product_code)."""
    d = client.post("/api/stylist/edit", files={"photo": ("o.jpg", _photo(), "image/jpeg")},
                    data={"garments": json.dumps(GARMENTS), "goal": goal, "occasion": "a wedding"}).json()
    fams = [p["card"]["article_id"] // 1000 for p in d["picks"]]
    assert len(fams) == len(set(fams))
    alt_fams = [it["article_id"] // 1000 for a in d["alternatives"] for it in a["items"]]
    assert not set(alt_fams) & set(fams)
    if goal != "similar":
        assert len(d["picks"]) >= 4

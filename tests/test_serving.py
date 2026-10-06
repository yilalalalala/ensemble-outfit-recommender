"""Track B serving: API contract, fallbacks, availability, evidence-gated reasons, uploads,
failure injection, cache invalidation and bundle reload -- on a synthetic bundle (no H&M data)."""
import io
import json
import shutil
from datetime import date

import duckdb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ensemble.serving import fixture
from ensemble.serving.runtime import PERSONAL, Bundle, BundleError, Recommender, personal_matrix


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory):
    return fixture.build(tmp_path_factory.mktemp("bundle") / "fixture-v1")


@pytest.fixture()
def api(bundle_dir):
    from ensemble.api import app as A
    A.STATE.load(bundle_dir)
    with TestClient(A.app) as c:
        A.STATE.load(bundle_dir)          # startup may have loaded the real bundle; pin the fixture
        A.STATE.rec.set_unavailable([])
        yield c, A


def ctl(c, **params):
    return c.get("/api/v2/complete-the-look", params=params)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def test_health_readiness_and_meta(api):
    c, _ = api
    assert c.get("/healthz").json()["status"] == "ok"
    r = c.get("/readyz").json()
    assert r["ready"] and r["bundle_version"] == "fixture-v1"
    m = c.get("/api/v2/meta").json()
    assert m["catalog_version"] == "fixturecat" and m["upload_limits"]["max_bytes"] == 8 * 1024 * 1024


def test_anonymous_request_is_compatibility_ordered_with_versions_and_request_id(api):
    c, _ = api
    r = c.get("/api/v2/complete-the-look", params={"anchor": 100, "slots": "shoes,lower"},
              headers={"X-Request-ID": "abc123"})
    assert r.status_code == 200 and r.headers["X-Request-ID"] == "abc123"
    body = r.json()
    assert body["request_id"] == "abc123" and not body["personalized"] and not body["customer_known"]
    assert {"model_version", "catalog_version", "profile_version", "availability_version", "bundle_version"} <= set(body)
    for m in body["modules"]:
        assert m["fallback_level"] == "anchor_pool" and not m["personalized"] and 1 <= len(m["items"]) <= 8
        for it in m["items"]:
            assert it["target_slot"] == m["slot"] and it["card"]["article_id"] == it["article_id"]
            assert it["provenance"] in it["evidence_types"] or it["provenance"] == "other"


def test_personalized_request_reorders_the_pool(api):
    c, _ = api
    anon = ctl(c, anchor=100, slots="shoes", k=6).json()["modules"][0]
    pers = ctl(c, anchor=100, slots="shoes", k=6, customer=1).json()
    m = pers["modules"][0]
    assert pers["personalized"] and pers["customer_known"] and m["personalized"]
    # customer 1 buys product type 13 (share 0.8 > 0.2): tree 2 lifts it by +3. The fixture's only
    # type-13 shoe is article 121, which the anonymous (compatibility) order does not put first.
    assert [it["article_id"] for it in m["items"]] != [it["article_id"] for it in anon["items"]]
    assert m["items"][0]["article_id"] == 121 and anon["items"][0]["article_id"] != 121
    assert m["items"][0]["card"]["product_type_name"] == "Type 13"


def test_unknown_customer_falls_back_to_compatibility(api):
    c, _ = api
    a = ctl(c, anchor=100, slots="shoes").json()
    b = ctl(c, anchor=100, slots="shoes", customer=999).json()
    assert not b["personalized"] and not b["customer_known"]
    assert [i["article_id"] for i in a["modules"][0]["items"]] == [i["article_id"] for i in b["modules"][0]["items"]]


def test_error_schema(api):
    c, _ = api
    for params, status, code in (({"anchor": 999999}, 404, "unknown_article"),
                                 ({"anchor": 100, "slots": "hats"}, 422, "invalid_slot"),
                                 ({"anchor": 100, "slots": "upper"}, 422, "invalid_slot")):
        r = ctl(c, **params)
        assert r.status_code == status, r.text
        e = r.json()["error"]
        assert e["code"] == code and e["request_id"] == r.headers["X-Request-ID"]
    assert ctl(c, anchor=100, k=0).status_code == 422        # typed query validation


# ---------------------------------------------------------------------------
# Fallback ladder, availability, reasons
# ---------------------------------------------------------------------------

def test_fallback_ladder_levels(api):
    c, _ = api
    # 104/105 are live shoes: no pool of their own, so every slot falls back.
    sib = ctl(c, anchor=141, slots="shoes")               # non-live upper; no live colourway either
    assert sib.json()["modules"][0]["fallback_level"] in ("visual_neighbor", "style_sibling")
    vis = ctl(c, anchor=140, slots="shoes").json()["modules"][0]
    assert vis["fallback_level"] == "visual_neighbor" and vis["source_anchor"] == 109
    pop = ctl(c, anchor=104, slots="accessories").json()["modules"][0]   # shoe anchor: no pools at all
    assert pop["fallback_level"] in ("visual_neighbor", "slot_popularity")
    empty = ctl(c, anchor=100, slots="swimwear").json()["modules"][0]
    assert empty["fallback_level"] == "slot_popularity" and empty["items"] == []


def test_style_sibling_fallback(bundle_dir):
    b = Bundle(bundle_dir)
    rec = Recommender(b)
    # Article 100 and 101 share a style; drop 100's pools from the key index to simulate a non-pooled anchor.
    keep = b.key_anchor != 100
    b.key_anchor, b.key_slot, b.key_offset, b.key_count = (x[keep] for x in (b.key_anchor, b.key_slot,
                                                                                b.key_offset, b.key_count))
    m = rec.module(100, "shoes", None, 5)
    assert m["fallback_level"] == "style_sibling" and m["source_anchor"] == 101


def test_availability_filter_and_cache_invalidation(api):
    c, A = api
    first = ctl(c, anchor=100, slots="shoes", k=4).json()
    v1 = first["availability_version"]
    gone = first["modules"][0]["items"][0]["article_id"]
    again = ctl(c, anchor=100, slots="shoes", k=4).json()
    assert again["modules"][0]["cache"] == "hit"
    r = c.post("/admin/availability", json={"unavailable": [gone]}).json()
    assert r["availability_version"] != v1
    after = ctl(c, anchor=100, slots="shoes", k=4).json()
    assert after["modules"][0]["cache"] == "miss" and after["availability_version"] == r["availability_version"]
    assert gone not in [i["article_id"] for i in after["modules"][0]["items"]]
    assert len(after["modules"][0]["items"]) == 4           # back-filled, still a full module


def test_reasons_are_gated_on_row_evidence(api):
    c, _ = api
    body = ctl(c, anchor=100, k=8, customer=1).json()
    n = 0
    for m in body["modules"]:
        for it in m["items"]:
            ev, keys = it["evidence"], {r["key"] for r in it["reasons"]}
            if "co_purchase" in keys:
                assert ev.get("a_co", 0) >= 1
            if "style" in keys:
                assert ev.get("s_co", 0) >= 1
            if "visual" in keys:
                assert ev.get("src_two_tower") == 1 or ev.get("src_two_tower_content") == 1
            if any("more often than chance" in r["text"] for r in it["reasons"]):
                assert ev["a_co"] >= 5
            if "colour" in keys:
                assert ev.get("same_colour_master") == 1
            if "price" in keys:
                assert ev.get("price_tier_diff") == 0
            assert {r["evidence_type"] for r in it["reasons"]} <= set(it["evidence_types"]) | {"attribute_match"}
            n += 1
    assert n > 0


# ---------------------------------------------------------------------------
# Uploads and degraded visual paths
# ---------------------------------------------------------------------------

def png(w=64, h=64) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.parametrize("data,ctype,status,code", [
    (b"not an image", "image/png", 400, "malformed_image"),
    (b"GIF89a....", "image/gif", 415, "unsupported_media_type"),
    (b"", "image/png", 400, "empty_upload"),
    ("tiny", "image/png", 422, "image_dimensions"),
])
def test_upload_validation(api, data, ctype, status, code):
    c, _ = api
    if data == "tiny":
        data = png(8, 8)
    r = c.post("/api/v2/visual-search", files={"photo": ("x", data, ctype)}, data={"category": "top"})
    assert r.status_code == status, r.text
    assert r.json()["error"]["code"] == code


def test_oversized_upload_rejected(api, monkeypatch):
    c, _ = api
    from ensemble.serving import visual
    monkeypatch.setitem(visual.LIMITS, "max_bytes", 100)
    r = c.post("/api/v2/visual-search", files={"photo": ("x", png(), "image/png")}, data={"category": "top"})
    assert r.status_code == 413 and r.json()["error"]["code"] == "upload_too_large"


def test_visual_search_with_injected_encoder_and_degraded_mode(api, monkeypatch, bundle_dir):
    c, A = api
    vs = A.STATE.visual
    target = int(np.load(bundle_dir / "visual" / "live_ids.npy")[3])          # a live upper article? any slot
    vec = np.asarray(A.STATE.rec.bundle.visual_vector(target), dtype=np.float32)
    monkeypatch.setattr(vs, "_load", lambda: setattr(vs, "state", "ready"))
    monkeypatch.setattr(vs, "_encode", lambda img, box: vec)
    slot = A.STATE.rec.bundle.catalog.at[target, "slot"]
    category = {"upper": "top", "lower": "bottom", "shoes": "shoes", "accessories": "belt"}[slot]
    if category == "belt":
        pytest.skip("fixture accessories have no belt product type")
    r = c.post("/api/v2/visual-search", files={"photo": ("x", png(), "image/png")},
               data={"category": category, "box": "0.1,0.1,0.9,0.9", "k": "3"}).json()
    assert r["mode"] == "crop" and r["degraded"] and r["matches"][0]["article_id"] == target
    # Model cannot load -> explicit 503, never a crash.
    monkeypatch.setattr(vs, "_load", lambda: (setattr(vs, "state", "unavailable"), setattr(vs, "reason", "no weights")))
    r = c.post("/api/v2/visual-search", files={"photo": ("x", png(), "image/png")}, data={"category": "top"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "visual_model_unavailable"


def test_outfit_complete_does_not_invent_a_gap(api, monkeypatch):
    c, A = api
    vs = A.STATE.visual
    vec = np.asarray(A.STATE.rec.bundle.visual_vector(101), dtype=np.float32)
    monkeypatch.setattr(vs, "_load", lambda: setattr(vs, "state", "ready"))
    monkeypatch.setattr(vs, "_encode", lambda img, box: vec)
    garments = [{"category": "top"}, {"category": "bottom"}, {"category": "shoes"}, {"category": "bag"}]
    r = c.post("/api/v2/outfit", files={"photo": ("x", png(), "image/png")},
               data={"garments": json.dumps(garments), "anchor": "100"}).json()
    assert r["outfit_complete"] and r["missing_slots"] == [] and r["modules"] == []
    r = c.post("/api/v2/outfit", files={"photo": ("x", png(), "image/png")},
               data={"garments": json.dumps(garments), "anchor": "100", "slots": "shoes"}).json()
    assert [m["slot"] for m in r["modules"]] == ["shoes"]
    r = c.post("/api/v2/outfit", files={"photo": ("x", png(), "image/png")},
               data={"garments": json.dumps(garments[:1]), "anchor": "100"}).json()
    assert r["missing_slots"] == ["lower", "shoes", "accessories"] and not r["outfit_complete"]


def test_outfit_without_garments_and_without_detector_degrades_cleanly(api, monkeypatch):
    c, _ = api
    monkeypatch.setenv("ENSEMBLE_LLM", "claude")          # paid backend is refused by the API
    r = c.post("/api/v2/outfit", files={"photo": ("x", png(), "image/png")})
    assert r.status_code == 503 and r.json()["error"]["code"] == "detector_unavailable"


def test_bundle_without_visual_index_reports_visual_unavailable(tmp_path):
    from ensemble.api import app as A
    d = fixture.build(tmp_path / "novis", with_visual=False, version="fixture-novis")
    A.STATE.load(d)
    with TestClient(A.app) as c:
        A.STATE.load(d)
        r = c.post("/api/v2/visual-search", files={"photo": ("x", png(), "image/png")}, data={"category": "top"})
        assert r.status_code == 503 and r.json()["error"]["code"] == "visual_model_unavailable"
        assert ctl(c, anchor=100, slots="shoes").status_code == 200       # CTL by id unaffected


# ---------------------------------------------------------------------------
# Failure injection: missing / corrupt artifacts, readiness, reload
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("damage", ["missing_model", "corrupt_manifest", "truncated_pool", "schema_mismatch",
                                    "wrong_format", "no_bundle"])
def test_bad_bundles_are_rejected(bundle_dir, tmp_path, damage):
    d = tmp_path / "b"
    shutil.copytree(bundle_dir, d)
    if damage == "missing_model":
        (d / "model_personalized.json").unlink()
    elif damage == "corrupt_manifest":
        (d / "manifest.json").write_text("{not json")
    elif damage == "truncated_pool":
        (d / "pool_feat.npy").write_bytes((d / "pool_feat.npy").read_bytes()[:-64])
    elif damage == "schema_mismatch":
        s = json.loads((d / "feature_schema.json").read_text())
        s["pool_features"] = s["pool_features"][::-1]
        (d / "feature_schema.json").write_text(json.dumps(s))
    elif damage == "wrong_format":
        m = json.loads((d / "manifest.json").read_text())
        m["format"] = 99
        (d / "manifest.json").write_text(json.dumps(m))
    elif damage == "no_bundle":
        d = tmp_path / "empty"
        d.mkdir()
    with pytest.raises(BundleError):
        Bundle(d)


def test_sha256_verification_catches_same_size_corruption(bundle_dir, tmp_path):
    d = tmp_path / "b"
    shutil.copytree(bundle_dir, d)
    raw = bytearray((d / "pool_compat.npy").read_bytes())
    raw[-1] ^= 0xFF
    (d / "pool_compat.npy").write_bytes(bytes(raw))
    Bundle(d)                                       # size check passes ...
    with pytest.raises(BundleError):
        Bundle(d, verify_hashes=True)               # ... the full hash check does not


def test_not_ready_then_reload_and_failed_reload_keeps_serving(bundle_dir, tmp_path):
    from ensemble.api import app as A
    with TestClient(A.app) as c:
        A.STATE.rec, A.STATE.error = None, "simulated: no bundle"
        assert c.get("/readyz").status_code == 503
        r = ctl(c, anchor=100)
        assert r.status_code == 503 and r.json()["error"]["code"] == "not_ready"
        ok = c.post("/admin/reload", json={"path": str(bundle_dir)})
        assert ok.status_code == 200 and c.get("/readyz").json()["ready"]
        ctl(c, anchor=100, slots="shoes")
        bad = tmp_path / "bad"
        bad.mkdir()
        r = c.post("/admin/reload", json={"path": str(bad)})
        assert r.status_code == 409 and r.json()["error"]["code"] == "bundle_invalid"
        assert c.get("/readyz").json()["bundle_version"] == "fixture-v1"      # old bundle still serving
        # A swap to a new version invalidates cached modules (the version is in every key).
        v2 = fixture.build(tmp_path / "v2", version="fixture-v2")
        c.post("/admin/reload", json={"path": str(v2)})
        r = ctl(c, anchor=100, slots="shoes").json()
        assert r["bundle_version"] == "fixture-v2" and r["modules"][0]["cache"] == "miss"


def test_metrics_and_structured_logs_do_not_leak_identifiers(api, caplog):
    c, A = api
    import logging
    with caplog.at_level(logging.INFO, logger="ensemble.requests"):
        A.LOG.propagate = True
        try:
            ctl(c, anchor=100, slots="shoes", customer=1)
        finally:
            A.LOG.propagate = False
    rec = [r for r in caplog.records if getattr(r, "endpoint", "") == "/api/v2/complete-the-look"][-1]
    assert rec.customer_ref and rec.customer_ref != "1" and rec.fallback_levels == ["anchor_pool"]
    assert rec.bundle_version == "fixture-v1" and rec.result_count == 8
    text = c.get("/metrics").text
    assert "ensemble_requests" in text and "ensemble_latency_ms_bucket" in text and "ensemble_ready 1" in text
    j = c.get("/api/v2/metrics").json()
    assert j["ready"] and "hits" in j["cache"]


def test_admin_endpoints_are_local_only(api):
    from ensemble.api.app import RequestError, _local_only

    class R:
        client = type("C", (), {"host": "10.1.2.3"})()
    with pytest.raises(RequestError):
        _local_only(R())


# ---------------------------------------------------------------------------
# Online personalization features == the production SQL (no H&M data needed)
# ---------------------------------------------------------------------------

def test_online_personal_features_match_the_production_sql():
    from ensemble.completion import features as FE
    from ensemble.config import Config, load_config
    from ensemble.data.splits import Week
    from ensemble.serving.bundle import profile_sql
    from ensemble.serving.runtime import decode_profile, encode_profile
    week = Week(date(2020, 9, 23), date(2020, 9, 29))
    con = duckdb.connect()
    con.execute("""CREATE TABLE articles AS SELECT * FROM (VALUES
        (1, 'upper', 11, 101, 1, 3, 5, 7, '09', 2, 21, 0, 1, FALSE),
        (2, 'lower', 12, 102, 2, 3, 6, 7, '10', 3, 22, 0, 1, FALSE),
        (3, 'shoes', 13, 103, 2, 4, 6, 8, '09', 2, 23, 1, 2, FALSE),
        (4, 'shoes', 13, 103, 2, 4, 6, 8, '11', 4, 23, 1, 2, FALSE),
        (5, 'accessories', 14, 104, 3, 5, 7, 9, NULL, 5, 24, 1, 2, TRUE))
        t(article_id, slot, product_code, product_type_no, graphical_appearance_no, department_no,
          section_no, garment_group_no, colour_group_code, perceived_colour_master_id, perceived_colour_value_id,
          index_group_no, garment_group, is_jewellery)""")
    con.execute("CREATE TABLE transactions (t_dat DATE, customer_idx INT, article_id INT, price FLOAT, sales_channel_id TINYINT)")
    rows = [("2020-09-20", 7, 1, 0.02), ("2020-09-20", 7, 2, 0.03), ("2020-09-01", 7, 3, 0.05),
            ("2020-06-01", 7, 3, 0.05), ("2019-01-01", 7, 5, 0.01), ("2020-09-22", 8, 4, 0.04),
            ("2020-09-25", 7, 4, 0.9)]                       # the last row is after the cutoff
    con.executemany("INSERT INTO transactions VALUES (?, ?, ?, ?, 2)", rows)
    cfg = load_config()
    cfg = Config({**cfg, "track_b": {**dict(cfg.track_b), "customer_weeks": 104}})
    q = pd.DataFrame({"qid": [0, 1, 2], "customer_idx": [7, 8, 9], "anchor": [1, 1, 2],
                      "target_slot": ["shoes", "shoes", "accessories"]})
    con.register("_q", q)
    con.execute("CREATE TEMP TABLE _tbq AS SELECT * FROM _q")
    FE.register_attrs(con)
    FE._customer_tables(con, week, cfg)
    con.execute("""CREATE TEMP TABLE _tb_price AS SELECT article_id, avg(price) AS mean_price, 1 AS price_tier
                   FROM transactions WHERE t_dat < DATE '2020-09-23' GROUP BY 1""")
    con.execute("""CREATE TEMP TABLE _tb_assoc AS SELECT * FROM (VALUES (1, 'shoes', 3, 0.37, 4), (1, 'shoes', 4, -0.1, 2))
                   t(src, dst_slot, dst, npmi, co)""")
    con.execute("CREATE TEMP TABLE _tb_keysim AS SELECT * FROM (VALUES (1, 'shoes', 3, 0.5::FLOAT)) t(anchor, target_slot, article_id, tt_pair_sim)")
    con.execute("CREATE TEMP TABLE u AS SELECT * FROM (VALUES (0, 3), (0, 4), (1, 3), (1, 4), (2, 5)) t(qid, article_id)")
    off = con.execute(f"""SELECT u.qid, u.article_id, {FE.PERSONAL_SQL} 0 AS _end
        FROM u JOIN _tbq q USING (qid) JOIN _tb_attrs ca_attr ON ca_attr.article_id = u.article_id
        LEFT JOIN _tb_price cp ON cp.article_id = u.article_id
        LEFT JOIN _tb_assoc a ON a.src = q.anchor AND a.dst_slot = q.target_slot AND a.dst = u.article_id
        LEFT JOIN _tb_keysim ks ON ks.anchor = q.anchor AND ks.target_slot = q.target_slot AND ks.article_id = u.article_id
        {FE.PERSONAL_JOINS} ORDER BY u.qid, u.article_id""").df()
    off = FE.downcast(off.drop(columns=["_end"]))
    profiles = {int(c): decode_profile(encode_profile(json.loads(j)))
                for c, j in con.execute(profile_sql(week.start, 104)).fetchall()}
    attrs = con.execute("""SELECT article_id, product_code, product_type_no, department_no, section_no,
                                  garment_group_no, TRY_CAST(colour_group_code AS INTEGER) AS colour_group_code,
                                  perceived_colour_master_id FROM articles""").df().set_index("article_id")
    price = con.execute("SELECT article_id, mean_price FROM _tb_price").df().set_index("article_id").mean_price
    for qid, g in off.groupby("qid"):
        qq = q.iloc[qid]
        arts = g.article_id.to_numpy()
        a = attrs.loc[arts]
        cand = {c: a[c].to_numpy() for c in a.columns}
        cand["article_id"], cand["cp_mean_price"] = arts, price.reindex(arts).to_numpy()
        aux = con.execute(f"""SELECT x.article_id, a.npmi, a.co, ks.tt_pair_sim FROM (SELECT unnest(?::INT[]) AS article_id) x
                              LEFT JOIN _tb_assoc a ON a.src = {qq.anchor} AND a.dst_slot = '{qq.target_slot}' AND a.dst = x.article_id
                              LEFT JOIN _tb_keysim ks ON ks.anchor = {qq.anchor} AND ks.target_slot = '{qq.target_slot}'
                                                      AND ks.article_id = x.article_id
                              ORDER BY x.article_id""", [arts.tolist()]).df()[["npmi", "co", "tt_pair_sim"]].to_numpy(np.float64)
        online = personal_matrix(profiles.get(int(qq.customer_idx)), cand, qq.target_slot, aux)
        offline = g[PERSONAL].to_numpy(np.float32)
        np.testing.assert_array_equal(np.isnan(online), np.isnan(offline))
        np.testing.assert_array_equal(np.nan_to_num(online), np.nan_to_num(offline))
    assert 9 not in profiles and profiles[7]["n"] == 5          # future row and unknown customer excluded

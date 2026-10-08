"""Public demo build (docs/CLAUDECODE_PUBLIC_DEMO_FULL_UI_PLAN.md, D-046).

Three things are checked here:

1. the build is the real frontend — ``sync`` is deterministic, publishes exactly what
   ``src/ensemble/api/static`` contains, and leaves no absolute URL that would break under the
   GitHub project subpath;
2. every route the frontend can reach has a published fixture of the right shape, and no published
   fixture points at a product, photo or page that is not published;
3. the headline evaluation numbers the published DS Studio shows are the canonical ones, unrounded
   and unrecomputed. The canonical source is the serving store's ``reports`` table, which is what the
   local DS Studio reads; the constants are asserted even where that store is absent.
"""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
from pathlib import Path

import pytest
import yaml

from ensemble.api.public_demo import demo_config, sync

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "ensemble" / "api" / "static"
CFG = demo_config()
BUILD = ROOT / CFG["out"]
FX = BUILD / CFG["fixtures"]
SERVING_DB = ROOT / "data" / "processed" / "serving.sqlite"

needs_build = pytest.mark.skipif(not (FX / "manifest.json").exists(),
                                 reason="the public demo fixtures are not built (make public-demo)")

# A fixture the build wrote, by name. Anything else in the directory (an editor backup, a file-sync
# agent's "name 2.json" copy) is not part of the build and is neither read nor counted: the build is
# what `manifest.json` and `ctl/index.json` say it is.
FIXTURE_NAME = re.compile(r"^\d+(-\d+)?$")
PHOTO_NAME = re.compile(r"^\d+\.jpg$")


def fixtures(d: Path, pattern: re.Pattern = FIXTURE_NAME) -> list[Path]:
    return sorted(p for p in d.glob("*.json") if pattern.match(p.stem))


def read(p: Path):
    return json.loads(p.read_text())


@pytest.fixture(scope="module")
def manifest():
    return read(FX / "manifest.json")


@pytest.fixture(scope="module")
def metrics():
    return read(FX / "metrics.json")


# ---------------------------------------------------------------- 1. the build is the real frontend
def test_sync_is_deterministic_and_publishes_only_the_real_frontend(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    first = sync(CFG, a)
    second = sync(CFG, a)                       # a second run must change nothing
    assert second["written"] == [] and second["removed"] == []
    sync(CFG, b)
    files_a = {p.relative_to(a): p.read_bytes() for p in a.rglob("*") if p.is_file()}
    files_b = {p.relative_to(b): p.read_bytes() for p in b.rglob("*") if p.is_file()}
    assert files_a == files_b, "the same frontend must produce the same build"
    # Every frontend file is published, byte for byte, under the same relative path.
    for src in STATIC.rglob("*"):
        if src.is_file() and src.name != "index.html":
            assert files_a[Path("static") / src.relative_to(STATIC)] == src.read_bytes(), src
    assert set(first["files"]) == {str(p) for p in files_a}


def test_sync_removes_files_the_frontend_no_longer_has(tmp_path):
    sync(CFG, tmp_path)
    stale = tmp_path / "static" / "js" / "gone.js"
    stale.write_text("// left over from an earlier frontend")
    assert "static/js/gone.js" in sync(CFG, tmp_path)["removed"]
    assert not stale.exists()


def test_published_page_declares_demo_mode_and_uses_relative_urls(tmp_path):
    sync(CFG, tmp_path)
    html = (tmp_path / "index.html").read_text()
    assert f'<meta name="ensemble-demo" content="{CFG["fixtures"]}/">' in html
    for bad in ('href="/static', 'src="/static', '="/api', '="/images', '="/readyz'):
        assert bad not in html, bad
    # Every local asset the page references exists in the build, at a relative path.
    for url in re.findall(r'(?:href|src)="([^"#:]+)"', html):
        assert not url.startswith("/"), url
        assert (tmp_path / url).exists(), url
    assert "github.com/yilalalalala/ensemble-outfit-recommender" in html


def test_real_app_keeps_absolute_api_paths_and_no_demo_mode():
    """The production FastAPI behaviour is unchanged: the served page never declares demo mode and
    the frontend still calls the API at the server root."""
    index = (STATIC / "index.html").read_text()
    assert "ensemble-demo" not in index
    js = "\n".join(p.read_text() for p in sorted((STATIC / "js").glob("*.js")))
    for path in ["/api/customers", "/api/home/", "/api/v2/complete-the-look", "/readyz"]:
        assert path in js, path


def test_demo_mode_is_declared_by_the_build_not_guessed():
    src = (STATIC / "js" / "demo.js").read_text()
    assert 'meta[name="ensemble-demo"]' in src
    for guess in ("github.io", "location.hostname", "location.host", "document.domain"):
        assert guess not in src, guess


@needs_build
def test_no_network_request_can_leave_the_published_build():
    """In demo mode every request is answered from a fixture. Statically: the published page and the
    published fixtures carry no absolute URL, and the only fetch() call sites in the frontend are the
    three in core.js (all behind the demo switch) and the fixture loader in demo.js. The runtime check
    is in scripts/ui/public_journeys.js, which fails on any request that leaves the site."""
    for p in [FX / "manifest.json", FX / "customers.json", FX / "families.json", FX / "assistant.json",
              FX / "photo.json", FX / "metrics.json", FX / "serving.json", BUILD / "index.html"] \
             + fixtures(FX / "ctl")[:50] + fixtures(FX / "product")[:50] + fixtures(FX / "home"):
        # serving.json holds the bundle's own telemetry, whose counters are *labelled* by endpoint
        # path ("/api/v2/complete-the-look"). Those labels are table cells, never request targets.
        if p.name == "serving.json":
            continue
        text = p.read_text()
        for bad in ('"/api/', '"/images/', '"/static/', '"/readyz"', "http://", "images/../"):
            assert bad not in text, f"{p.relative_to(BUILD)} carries {bad}"
    js = {p.name: p.read_text() for p in (BUILD / "static" / "js").glob("*.js")}
    for name, src in js.items():
        if name not in ("core.js", "demo.js"):
            assert "fetch(" not in src, f"{name} must go through core.js"
    core, demo = js["core.js"], js["demo.js"]
    assert core.count("fetch(") == 3                        # api(), probe(), logEvent()
    assert "if (DEMO) {" in core and "return await demoRequest(path, opts);" in core
    assert "if (DEMO) return demoRequestRaw(path);" in core
    assert "if (DEMO) return;   // documented no-op" in core
    assert demo.count("fetch(") == 1 and "fetch(`${ROOT}${DEMO}${name}`)" in demo


# ---------------------------------------------------------------- 2. fixtures: every route, closed set
@needs_build
def test_every_published_route_has_a_fixture_of_the_right_shape(manifest):
    customers = read(FX / "customers.json")
    assert [c["customer_idx"] for c in customers] == CFG["customers"]
    assert CFG["default_customer"] == customers[0]["customer_idx"]
    assert {c["segment"] for c in customers} == {"returning", "new"}
    for c in CFG["customers"]:
        home = read(FX / "home" / f"{c}.json")
        assert home["customer"]["customer_idx"] == c
        assert [m["id"] for m in home["modules"]] == ["for_you", "buy_again", "trending"]
        assert any(m["items"] for m in home["modules"])
        assert (FX / "explain" / f"{c}.json").exists()
    serving = read(FX / "serving.json")
    assert serving["readyz"]["ready"] is True
    assert serving["meta"]["bundle_version"] == serving["readyz"]["bundle_version"] == manifest["bundle_version"]
    assert {"counters", "histograms"} <= set(serving["metrics"])
    assert len(fixtures(FX / "product")) == manifest["product_fixtures"] == manifest["published_articles"]
    # The index is what the page consults, so it must name exactly the captures that exist: the page
    # never requests a Complete-the-Look file that was not published.
    index = read(FX / "ctl" / "index.json")
    assert sorted(index) == sorted(p.stem for p in fixtures(FX / "ctl"))
    assert len(index) == manifest["ctl_fixtures"]


@needs_build
def test_published_articles_are_a_closed_set(manifest):
    """Nothing a visitor can click leads to a product, photo or page that is not published."""
    published = {int(p.stem) for p in fixtures(FX / "product")}
    assert len(published) == manifest["published_articles"]
    images = {int(p.stem) for p in (BUILD / CFG["images"]).glob("*.jpg") if PHOTO_NAME.match(p.name)}
    assert images == published, sorted(published ^ images)[:10]

    def ids(obj):
        """Article ids a visitor can be shown. A tool-call argument inside a saved `trace` is not
        one: it records what the assistant asked for, not a product on the page."""
        if isinstance(obj, dict):
            out = [int(obj["article_id"])] if "article_id" in obj else []
            return out + [i for k, v in obj.items() if k != "trace" for i in ids(v)]
        return [i for v in obj for i in ids(v)] if isinstance(obj, list) else []

    named = [FX / n for n in ("customers.json", "families.json", "assistant.json", "photo.json")]
    for d in ("home", "explain", "product", "ctl"):
        named += fixtures(FX / d)
    for p in named:
        missing = sorted(set(ids(read(p))) - published)
        assert not missing, f"{p.relative_to(FX)} references unpublished articles {missing[:5]}"


@needs_build
def test_every_image_reference_is_relative_and_resolves():
    text = "\n".join((FX / n).read_text() for n in ("families.json", "assistant.json", "photo.json"))
    text += "\n".join(p.read_text() for d in ("home", "product", "ctl") for p in fixtures(FX / d)[:60])
    srcs = set(re.findall(r'"(images/\d+\.jpg)"', text))
    assert srcs
    for s in srcs:
        assert (BUILD / s).exists(), s
    assert read(FX / "photo.json")["photo"]["src"] == "photo.jpg"
    assert (BUILD / "photo.jpg").exists()


@needs_build
def test_complete_the_look_modules_are_the_live_response_filtered_not_padded():
    floor = CFG["catalog"]["min_module_items"]
    for p in fixtures(FX / "ctl")[:40]:
        d = read(p)
        assert d["modules"], p.name
        for m in d["modules"]:
            assert floor <= len(m["items"]) <= CFG["catalog"]["ctl_k"], (p.name, m["slot"])
            assert [i["rank"] for i in m["items"]] == list(range(1, len(m["items"]) + 1))
            assert len({i["article_id"] for i in m["items"]}) == len(m["items"])
            for it in m["items"]:
                assert {"provenance", "reasons", "evidence", "card"} <= set(it)
    # The default profile's own anchors are captured personalized, as localhost serves them.
    personal = [p for p in fixtures(FX / "ctl") if p.stem.endswith(f"-{CFG['default_customer']}")]
    assert personal, "no personalized Complete the Look was captured for the default profile"
    assert all(read(p)["personalized"] for p in personal)


@needs_build
def test_saved_assistant_turns_are_grounded_and_demonstrable():
    d = read(FX / "assistant.json")
    want = [q["id"] for q in CFG["assistant"]["questions"]]
    by_id = {q["id"]: q for q in CFG["assistant"]["questions"]}
    assert [e["id"] for e in d["examples"]] == want
    with_cards = 0
    for e in d["examples"]:
        assert e["answer"].strip()
        assert e["hallucinated"] == [], e["id"]
        assert isinstance(e["trace"], list)
        with_cards += bool(e["cards"])
        assert e["customer"] == by_id[e["id"]].get("customer", CFG["assistant"]["customer"]), e["id"]
        for c in e["cards"]:
                # Every product shown under an answer is one the answer actually cites, in
                # whichever form the model wrote it ([[id]] or a bare id): agent.py grounds both.
                assert str(c["article_id"]) in e["answer"], (e["id"], c["article_id"])
    assert with_cards >= 4, "the saved conversation must demonstrate grounded recommendations"
    assert {e["id"] for e in d["examples"] if e["suggested"]} == set(CFG["assistant"]["suggested"])
    pairs = [(e["id"], e["followup_of"]) for e in d["examples"] if e["followup_of"]]
    assert pairs and all(f in want for _, f in pairs)


@needs_build
def test_saved_photo_example_is_openly_licensed_and_carries_both_services():
    d = read(FX / "photo.json")
    assert d["visual_search"]["matches"], "the saved visual search must return products"
    assert any(g["matches"] for g in d["snap"]["garments"])
    assert d["snap"]["complete_the_look"]
    credit = d["photo"]["credit"]
    assert "Photo:" in credit and ("CC" in credit or "public domain" in credit.lower())
    assert d["photo"]["credit_url"].startswith("https://commons.wikimedia.org/")
    assert len(d["photo"]["box"]) == 4 and all(0 <= v <= 1 for v in d["photo"]["box"])


@needs_build
def test_the_published_build_carries_no_private_photo_and_no_generated_product_image():
    """Only the one openly licensed outfit photo and the catalogue photography are published."""
    assert (BUILD / "photo.jpg").exists()
    assert not [p.name for p in BUILD.glob("*.jpg") if p.name.startswith("photo") and p.name != "photo.jpg"]
    assert not (BUILD / "assets").exists(), "the old hand-written portfolio assets must be gone"
    index = {str(a) for a in read(FX / "families.json")["articles"]}
    for p in (BUILD / CFG["images"]).glob("*.jpg"):
        if PHOTO_NAME.match(p.name):
            assert p.stem in index, p   # every product image is a published catalogue article


# ---------------------------------------------------------------- 3. results are the canonical ones
BEST_BASELINE = "repeat_purchase+popularity_by_age"


def f5(x):
    return f"{x:.5f}"


def pct1(x):
    return f"{'+' if x > 0 else ''}{x * 100:.1f}%"


def headline_from(metrics: dict) -> dict:
    """The four values the DS Studio overview shows, derived exactly as studio.js derives them."""
    rt = metrics["m3_ranker_test"]["metrics"]
    rv = metrics["m3_ranker_val"]["metrics"]
    bt = metrics["m1_baselines_test"]
    tb = metrics.get("m4_track_b_test") or metrics["m4_track_b_val"]
    lift = (rt["map@12"] - bt[BEST_BASELINE]["map@12"]) / bt[BEST_BASELINE]["map@12"]
    return {"ranker_test_map12": f5(rt["map@12"]), "ranker_test_lift_vs_best_baseline": pct1(lift),
            "ranker_val_map12": f5(rv["map@12"]), "track_b_models_compared": len(tb["results"]),
            "stored_reports": len(metrics)}


@needs_build
def test_published_headline_metrics_are_the_configured_canonical_values(metrics):
    assert headline_from(metrics) == {k: v for k, v in CFG["headline"].items()}


@needs_build
def test_published_headline_metrics_are_the_last_verified_results(metrics):
    """Hard-coded so the check still holds where the serving store is absent (CI)."""
    h = headline_from(metrics)
    assert h["ranker_test_map12"] == "0.03918"
    assert h["ranker_test_lift_vs_best_baseline"] == "+45.7%"
    assert h["ranker_val_map12"] == "0.03772"
    assert h["track_b_models_compared"] == 10
    assert h["stored_reports"] == 7
    # Unrounded, as stored: studio.js puts the full value in the cell's title attribute.
    assert metrics["m3_ranker_test"]["metrics"]["map@12"] == 0.03917510232956311
    assert metrics["m3_ranker_val"]["metrics"]["map@12"] == 0.03772064562082999


@needs_build
@pytest.mark.skipif(not SERVING_DB.exists(), reason="the local serving store is not present")
def test_published_reports_are_the_serving_stores_reports_unchanged(metrics):
    con = sqlite3.connect(f"file:{SERVING_DB}?mode=ro", uri=True)
    try:
        canonical = {name: json.loads(body) for name, body in con.execute("SELECT name, body FROM reports")}
    finally:
        con.close()
    assert metrics == canonical, "the published evaluation reports must equal the local ones exactly"


@needs_build
def test_results_are_labelled_offline_and_never_claim_live_traffic():
    studio = (BUILD / "static" / "js" / "studio.js").read_text()
    assert "All numbers are offline (historical data). A launch decision would need an online A/B test." in studio
    assert "captured when the published build was made" in studio
    assert "not public traffic" in studio
    # No checkout, in either surface.
    assert 'class="btn-cart" type="button" disabled>Checkout' in (BUILD / "index.html").read_text()


@needs_build
def test_the_workflow_still_deploys_this_build():
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "pages.yml").read_text())
    upload = next(s for j in wf["jobs"].values() for s in j["steps"] if "upload-pages-artifact" in str(s.get("uses")))
    assert upload["with"]["path"] == CFG["out"]
    paths = wf[True]["push"]["paths"] if True in wf else wf["on"]["push"]["paths"]
    assert f"{CFG['out']}/**" in paths

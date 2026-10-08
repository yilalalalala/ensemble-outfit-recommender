"""Editorial UI (docs/CLAUDECODE_UI_EDITORIAL_REDESIGN_PLAN.md): static routing, the page's critical
semantics, and that the frontend still calls every API contract the previous UI used."""
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ensemble.api.app import app

STATIC = Path(__file__).resolve().parents[1] / "src" / "ensemble" / "api" / "static"
JS = sorted((STATIC / "js").glob("*.js"))


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def js_source() -> str:
    return "\n".join(p.read_text() for p in JS)


def test_index_and_static_assets_are_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    for path, ctype in [("/static/css/ensemble.css", "text/css"), ("/static/js/app.js", "javascript"),
                        ("/static/favicon.svg", "image/svg+xml"), ("/static/placeholder.svg", "image/svg+xml")]:
        res = client.get(path)
        assert res.status_code == 200, path
        assert ctype in res.headers["content-type"], (path, res.headers["content-type"])


def test_every_module_import_resolves(client):
    for p in JS:
        for name in re.findall(r'from\s+"\./([\w.]+)"', p.read_text()):
            assert (STATIC / "js" / name).exists(), f"{p.name} imports missing {name}"
            assert client.get(f"/static/js/{name}").status_code == 200


class _Tree(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.els = [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        el = {"tag": tag, "attrs": a, "parents": [dict(p) for p in self.stack], "text": ""}
        self.els.append(el)
        if tag not in ("meta", "link", "input", "img", "br", "path", "circle"):
            self.stack.append({"tag": tag, **a})

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i]["tag"] == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        if self.els:
            self.els[-1]["text"] += data


@pytest.fixture(scope="module")
def tree():
    t = _Tree()
    t.feed((STATIC / "index.html").read_text())
    return t.els


def test_landmarks_and_centered_lowercase_wordmark(tree):
    mains = [e for e in tree if e["tag"] == "main"]
    assert len(mains) == 1 and mains[0]["attrs"]["id"] == "main"
    assert any(e["tag"] == "header" for e in tree) and any(e["tag"] == "footer" for e in tree)
    assert tree[0]["tag"] == "html" and tree[0]["attrs"].get("lang") == "en"
    wm = next(e for e in tree if e["attrs"].get("id") == "wordmark")
    assert wm["attrs"].get("aria-label") == "ensemble"
    assert any(p.get("class") == "masthead__inner" for p in wm["parents"])
    # Decorative letter spans spell the word and are hidden from assistive technology.
    spans = [e for e in tree if e["tag"] == "span" and any(p.get("id") == "wordmark" for p in e["parents"])]
    assert "".join(e["text"].strip() for e in spans) == "ensemble" and all(e["attrs"].get("aria-hidden") == "true" for e in spans)
    css = (STATIC / "css" / "ensemble.css").read_text()
    # Three-column masthead: the wordmark sits in the auto column, so it is centred whatever the side widths.
    assert re.search(r"\.masthead__inner\s*\{[^}]*grid-template-columns:\s*1fr auto 1fr", css)
    assert re.search(r"\.wordmark\s*\{[^}]*text-transform:\s*lowercase", css)
    assert "skip" in [e["attrs"].get("class") for e in tree if e["tag"] == "a"]


def test_experience_switch_is_separate_from_the_profile_selector(tree):
    switch = [e for e in tree if e["tag"] == "a" and "data-exp" in e["attrs"]]
    assert {e["attrs"]["data-exp"] for e in switch} == {"shop", "studio"}
    assert all(any(p.get("aria-label", "").startswith("Experience") for p in e["parents"]) for e in switch)
    profile = next(e for e in tree if e["attrs"].get("id") == "profile-btn")
    assert profile["tag"] == "button" and profile["attrs"].get("aria-haspopup") == "dialog"
    assert not any(p.get("aria-label", "").startswith("Experience") for p in profile["parents"])
    assert any(e["tag"] == "dialog" and e["attrs"].get("id") == "profile-dialog" for e in tree)


def test_no_remote_dependencies_or_build_artifacts():
    text = (STATIC / "index.html").read_text() + (STATIC / "css" / "ensemble.css").read_text() + js_source()
    assert not re.search(r"https?://", text), "no remote fonts, scripts or trackers"
    assert not list(STATIC.rglob("*.map")) and not (STATIC / "node_modules").exists()


def test_frontend_keeps_every_api_contract_and_event():
    src = js_source()
    for path in ["/api/customers", "/api/home/", "/api/product/", "/api/explain/", "/api/metrics", "/api/events",
                 "/api/visual-search", "/api/snap", "/api/assistant/session", "/api/assistant/", "/api/label/tasks",
                 '"/api/label"', "/api/v2/complete-the-look", "/readyz", "/api/v2/meta", "/api/v2/metrics", "/api/catalog/families"]:
        assert path in src, path
    for event in ['"impression"', '"click"', '"not_for_me"', '"add_to_cart"']:
        assert event in src, event
    for route in ['"product"', '"assistant"', '"ds"', '"label"', '"studio"', '"collections"']:
        assert route in src, f"route {route}"


def test_motion_respects_reduced_motion():
    css = (STATIC / "css" / "ensemble.css").read_text()
    assert "@media (prefers-reduced-motion: no-preference)" in css
    # All reveal / transition animation lives behind the no-preference query.
    head, _, rest = css.partition("@media (prefers-reduced-motion: no-preference)")
    assert "@keyframes" not in head and ".reveal" not in head


SHOPPER_MODULES = ["shop.js", "stylist.js", "cart.js", "catalog.js", "core.js", "app.js"]


def test_shopper_modules_never_explain_or_show_why_this():
    for name in ["shop.js", "stylist.js", "cart.js", "catalog.js"]:
        src = (STATIC / "js" / name).read_text()
        assert "/api/explain" not in src, name
        assert "Why this" not in src and "data-why" not in src, name
    html = (STATIC / "index.html").read_text()
    assert "why-dialog" not in html
    for banned in ("no prices", "no checkout", "Live outfit service unavailable"):
        assert banned not in html + js_source(), banned


def test_white_canvas_without_automatic_dark_mode_and_black_cart_buttons():
    css = (STATIC / "css" / "ensemble.css").read_text()
    assert "prefers-color-scheme: dark" not in css
    assert re.search(r"--canvas:\s*#ffffff", css)
    rule = re.search(r"\.btn-cart\s*\{([^}]*)\}", css).group(1)
    assert "background: #000" in rule and "color: #fff" in rule


def test_cart_drawer_and_masthead_control(tree):
    btn = next(e for e in tree if e["attrs"].get("id") == "cart-btn")
    assert btn["tag"] == "button" and btn["attrs"].get("aria-controls") == "cart-dialog"
    assert any(e["tag"] == "dialog" and e["attrs"].get("id") == "cart-dialog" for e in tree)

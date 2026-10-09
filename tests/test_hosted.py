"""Hosted shop (D-047): the GitHub Pages build of the real frontend calling the live Modal API.
No data or network needed."""
import importlib.util
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ensemble.api import app as app_module

ROOT = Path(__file__).resolve().parents[1]
API = "https://example--ensemble-recommender-web.modal.run"


def _builder():
    spec = importlib.util.spec_from_file_location("build_cloud_frontend", ROOT / "scripts" / "deploy" / "build_cloud_frontend.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pages_build_points_at_the_live_api_with_relative_assets(tmp_path):
    _builder().build(API, tmp_path)
    html = (tmp_path / "index.html").read_text()
    assert f'<meta name="ensemble-api-base" content="{API}">' in html
    assert 'name="ensemble-demo"' not in html                      # no frozen-fixture mode
    assert '"/static/' not in html and 'href="static/css/ensemble.css"' in html
    for js in ("app.js", "core.js", "shop.js", "stylist.js", "studio.js", "cart.js", "catalog.js"):
        assert (tmp_path / "static" / "js" / js).exists(), js
    assert not (tmp_path / "static" / "js" / "demo.js").exists()
    assert not (tmp_path / "demo").exists() and not (tmp_path / "images").exists()
    assert not [p for p in tmp_path.rglob("*") if re.search(r" [2-9]\.[^.]+$", p.name)]   # sync duplicates skipped
    assert (tmp_path / ".nojekyll").exists()


def test_pages_build_requires_https(tmp_path):
    with pytest.raises(SystemExit):
        _builder().build("http://insecure.example", tmp_path)


def test_published_portfolio_is_the_live_build():
    html = (ROOT / "portfolio" / "index.html").read_text()
    assert 'name="ensemble-api-base"' in html and 'name="ensemble-demo"' not in html


def test_frontend_has_no_frozen_demo_mode():
    static = ROOT / "src" / "ensemble" / "api" / "static" / "js"
    assert not (static / "demo.js").exists()
    for p in static.glob("*.js"):
        if re.search(r" [2-9]\.js$", p.name):
            continue
        assert "DEMO" not in p.read_text(), p.name


def test_hosted_api_refuses_the_local_label_tool(monkeypatch):
    monkeypatch.setattr(app_module, "HOSTED", True)
    c = TestClient(app_module.app)
    assert c.get("/api/label/tasks?round=2").status_code == 404
    assert c.get("/api/label/crop/0.jpg?round=2").status_code == 404
    assert c.post("/api/label", json={"task_id": "x", "relevant": [0], "round": 2}).status_code == 404


def test_image_urls_switch_to_the_object_store_when_configured(monkeypatch):
    assert app_module.image_url(922037001) == "/images/922037001.jpg" or app_module.IMAGE_BASE_URL
    monkeypatch.setattr(app_module, "IMAGE_BASE_URL", "https://pub-x.r2.dev/catalog")
    assert app_module.image_url(922037001) == "https://pub-x.r2.dev/catalog/092/0922037001.jpg"

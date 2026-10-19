"""Run the pure shop-logic tests under node: tests/js/catalog.test.mjs (product-family grouping, colour-variant
labels and swatches, USD formatting, the cart model) and tests/js/find.test.mjs (Find this piece: labels and the
category guess for a framed area)."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_catalog_js():
    r = subprocess.run(["node", str(ROOT / "tests" / "js" / "catalog.test.mjs")], capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_find_pieces_js():
    r = subprocess.run(["node", str(ROOT / "tests" / "js" / "find.test.mjs")], capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_stylist_chat_js():
    r = subprocess.run(["node", str(ROOT / "tests" / "js" / "stylist.test.mjs")], capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr

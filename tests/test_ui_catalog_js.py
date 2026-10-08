"""Run the pure shop-logic tests (tests/js/catalog.test.mjs) under node: product-family grouping,
colour-variant labels and swatches, USD formatting and the cart model."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_catalog_js():
    r = subprocess.run(["node", str(ROOT / "tests" / "js" / "catalog.test.mjs")], capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr

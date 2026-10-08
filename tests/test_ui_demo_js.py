"""Run the public demo data adapter's unit tests (tests/js/demo.test.mjs) under node: fixture routing,
relative URLs under a project subpath, saved assistant turns and the backends that need a server."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.skipif(not (ROOT / "portfolio" / "demo" / "customers.json").exists(),
                    reason="the public demo fixtures are not built (make public-demo)")
def test_demo_js():
    r = subprocess.run(["node", str(ROOT / "tests" / "js" / "demo.test.mjs")],
                       capture_output=True, text=True, cwd=ROOT, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr

"""Build the real (non-fixture) frontend for GitHub Pages.

Only source frontend files are copied.  The generated index declares the Modal
API URL, so photo uploads, recommendations and assistant turns are all live.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "ensemble" / "api" / "static"
DUPLICATE = re.compile(r" [2-9]\.[^.]+$")


def build(api_base: str, out: Path) -> None:
    api_base = api_base.rstrip("/")
    if not api_base.startswith("https://"):
        raise SystemExit("--api-base must be an https URL")
    (out / "static").mkdir(parents=True, exist_ok=True)
    for src in STATIC.rglob("*"):
        # Skip local sync duplicates such as "demo 2.js": they are not part of the app.
        if src.is_file() and src.name != "index.html" and not DUPLICATE.search(src.name):
            dst = out / "static" / src.relative_to(STATIC)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    html = (STATIC / "index.html").read_text()
    meta = f'<meta name="ensemble-api-base" content="{api_base}">'
    html = html.replace("<head>\n", "<head>\n<!-- Live Modal backend; no frozen fixtures. -->\n" + meta + "\n", 1)
    # GitHub Pages serves the site under /<repo>/, so asset links must be relative.
    html = html.replace('"/static/', '"static/')
    (out / "index.html").write_text(html)
    shutil.copy2(STATIC / "favicon.svg", out / "favicon.svg")
    (out / ".nojekyll").write_text("")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--api-base", required=True)
    p.add_argument("--out", type=Path, default=ROOT / "portfolio")
    args = p.parse_args()
    build(args.api_base, args.out)

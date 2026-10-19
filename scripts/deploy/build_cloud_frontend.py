"""Build the real (non-fixture) frontend for GitHub Pages.

Only source frontend files are copied.  The generated index declares the Modal
API URL, so photo uploads, recommendations and assistant turns are all live.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "ensemble" / "api" / "static"
DUPLICATE = re.compile(r" [2-9]\.[^.]+$")


def app_files() -> list[Path]:
    """The frontend's own files (local sync duplicates such as "demo 2.js" are not part of the app)."""
    return sorted(p for p in STATIC.rglob("*") if p.is_file() and p.name != "index.html" and not DUPLICATE.search(p.name))


def fingerprint(files: list[Path]) -> str:
    """A content hash of the frontend: it changes whenever any file changes."""
    h = hashlib.sha1()
    for f in files:
        h.update(str(f.relative_to(STATIC)).encode() + b"\0" + f.read_bytes())
    return h.hexdigest()[:10]


def build(api_base: str, out: Path) -> None:
    api_base = api_base.rstrip("/")
    if not api_base.startswith("https://"):
        raise SystemExit("--api-base must be an https URL")
    files = app_files()
    version = f"v-{fingerprint(files)}"
    # Cache busting: the files live under static/v-<hash>/, so after an update every file has a new address and no
    # browser (or the Pages CDN) can mix the new page with old cached modules. Relative module imports inherit the
    # versioned directory. Earlier builds are removed: older v-* folders and the unversioned copies of app files
    # (never anything else, such as local duplicates).
    static = out / "static"
    for old in static.glob("v-*"):
        if old.is_dir() and old.name != version:
            shutil.rmtree(old)
    for f in files:
        stale = static / f.relative_to(STATIC)
        if stale.is_file():
            stale.unlink()
    for src in files:
        dst = static / version / src.relative_to(STATIC)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    html = (STATIC / "index.html").read_text()
    meta = f'<meta name="ensemble-api-base" content="{api_base}">'
    html = html.replace("<head>\n", "<head>\n<!-- Live Modal backend; no frozen fixtures. -->\n" + meta + "\n", 1)
    # GitHub Pages serves the site under /<repo>/, so asset links must be relative (and versioned).
    html = html.replace('"/static/', '"static/').replace('"static/', f'"static/{version}/')
    (out / "index.html").write_text(html)
    shutil.copy2(STATIC / "favicon.svg", out / "favicon.svg")
    (out / ".nojekyll").write_text("")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--api-base", required=True)
    p.add_argument("--out", type=Path, default=ROOT / "portfolio")
    args = p.parse_args()
    build(args.api_base, args.out)

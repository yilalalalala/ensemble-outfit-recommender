"""Placeholder so the suite runs before any module exists.

Replaced at M0 by the ingestion integrity checks described in docs/DATA.md.
"""


def test_repository_layout() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for expected in ("docs/DESIGN.md", "docs/DATA.md", "docs/DECISIONS.md",
                     "scripts/download_data.sh", "src/ensemble"):
        assert (root / expected).exists(), f"missing {expected}"

"""Prototype merchandising layer for the shop UI: product families, colourways and USD prices.

**These prices are portfolio/demo merchandising prices, not source-dataset or historical H&M
prices.** The source catalogue has no US retail price, so a checked-in policy
(``configs/merch_pricing.yaml``) maps real product attributes to a price band, and a stable hash of
the real ``product_code`` picks a standard retail price point inside it. Nothing here feeds a model,
a feature or an evaluation.

A *product family* is the catalogue's own ``product_code`` (verified against the local data: every
``article_id // 1000 == int(product_code)``); colourways are the family's other articles. Products are
never grouped by name.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import yaml

POLICY_PATH = Path(__file__).resolve().parents[3] / "configs" / "merch_pricing.yaml"


@lru_cache(maxsize=1)
def policy(path: str | None = None) -> dict:
    return yaml.safe_load(Path(path or POLICY_PATH).read_text())


def band(row: dict, pol: dict | None = None) -> tuple[float, float]:
    pol = pol or policy()
    b = (pol["by_product_type"].get(row.get("product_type_name"))
         or pol["by_product_group"].get(row.get("product_group_name"))
         or pol["default"])
    scale = float(pol["by_index_group"].get(row.get("index_group_name"), 1.0))
    if "Premium" in str(row.get("department_name") or ""):
        scale *= float(pol["premium_multiplier"])
    return float(b[0]) * scale, float(b[1]) * scale


def price_usd(row: dict, pol: dict | None = None) -> float:
    """Deterministic prototype USD price for the family of ``row`` (needs ``product_code`` and the
    attributes used by the band). Same family -> same price, in every process."""
    pol = pol or policy()
    lo, hi = band(row, pol)
    points = [float(p) for p in pol["price_points"]]
    inside = [p for p in points if lo - 1e-9 <= p <= hi + 1e-9]
    if not inside:   # a band narrower than the ladder after scaling: nearest point to its centre
        mid = (lo + hi) / 2
        inside = [min(points, key=lambda p: abs(p - mid))]
    h = int(hashlib.sha256(str(row["product_code"]).encode()).hexdigest(), 16)
    return inside[h % len(inside)]

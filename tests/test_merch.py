"""Prototype merchandising prices (src/ensemble/api/merch.py): deterministic, banded, family-stable.
These are demo prices for the shop UI, not source-dataset prices."""
import subprocess
import sys
from pathlib import Path

from ensemble.api import merch

ROW = {"product_code": "0922037", "product_type_name": "Trousers", "product_group_name": "Garment Lower body",
       "index_group_name": "Ladieswear", "department_name": "Trousers & Skirt"}


def test_price_is_deterministic_across_calls_and_processes():
    p = merch.price_usd(ROW)
    assert p == merch.price_usd(dict(ROW))
    root = Path(__file__).resolve().parents[1]
    code = f"from ensemble.api import merch; print(merch.price_usd({ROW!r}))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root,
                         env={"PYTHONPATH": str(root / "src"), "PYTHONHASHSEED": "123"}, check=True)
    assert float(out.stdout) == p


def test_prices_use_retail_endings_and_stay_inside_the_band():
    pol = merch.policy()
    for code in (f"{i:07d}" for i in range(500)):
        for ptype, group in [("Trousers", "Garment Lower body"), ("T-shirt", "Garment Upper body"), ("Earring", "Accessories"),
                             ("Coat", "Garment Upper body"), ("Unseen type", "Unseen group")]:
            row = {**ROW, "product_code": code, "product_type_name": ptype, "product_group_name": group}
            p = merch.price_usd(row, pol)
            assert f"{p:.2f}".split(".")[1] in ("00", "50", "90", "99"), p
            lo, hi = merch.band(row, pol)
            assert lo - 1e-9 <= p <= hi + 1e-9, (row, p, lo, hi)


def test_same_family_attributes_give_one_price_regardless_of_colour():
    a = merch.price_usd({**ROW, "colour_group_name": "Beige"})
    b = merch.price_usd({**ROW, "colour_group_name": "Black"})
    assert a == b


def test_band_scaling_children_divided_premium_and_fallbacks():
    pol = merch.policy()
    adult = merch.band(ROW, pol)
    assert merch.band({**ROW, "index_group_name": "Baby/Children"}, pol)[1] < adult[1]
    assert merch.band({**ROW, "index_group_name": "Divided"}, pol)[1] < adult[1]
    assert merch.band({**ROW, "department_name": "Woven Premium"}, pol)[1] > adult[1]
    assert merch.band({**ROW, "product_type_name": "Nope", "product_group_name": "Nope"}, pol) == tuple(map(float, pol["default"]))
    assert merch.band({**ROW, "product_type_name": "Nope"}, pol) == tuple(map(float, pol["by_product_group"]["Garment Lower body"]))


def test_policy_documents_that_prices_are_not_source_prices():
    text = merch.POLICY_PATH.read_text()
    assert "NOT historical or source-dataset prices" in text
    assert "not source-dataset or historical" in (merch.__doc__ or "")

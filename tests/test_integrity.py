"""DATA.md "Integrity checks", asserted against the built database."""
import json

import pytest

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def report(cfg):
    path = cfg.path("reports") / "m0_integrity.json"
    if not path.exists():
        pytest.skip("run `make ingest` first")
    return json.loads(path.read_text())


def test_row_counts_reconcile(report):
    assert report["db_rows"] == report["source_rows"]


def test_no_orphans(report):
    assert report["orphan_articles"] == 0
    assert report["orphan_customers"] == 0
    assert report["duplicate_article_ids"] == 0


def test_no_unexpected_gaps(report):
    assert report["missing_days"] == []


def test_slots_cover_wearables(con):
    n_null = con.execute(
        "SELECT count(*) FROM articles WHERE slot IS NULL "
        "AND product_group_name IN ('Garment Upper body','Shoes','Accessories')"
    ).fetchone()[0]
    assert n_null == 0

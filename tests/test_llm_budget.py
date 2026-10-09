"""The AI spend cap (src/ensemble/llm/client.py): monthly when configured, and a reached cap is a clean 503."""
import json

import pytest
from fastapi.testclient import TestClient

from ensemble.llm import client


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "SPEND_FILE", tmp_path / "spend.json")
    monkeypatch.setattr(client, "SPEND_DICT", None)
    return tmp_path / "spend.json"


def test_monthly_cap_counts_only_this_month(ledger, monkeypatch):
    monkeypatch.setattr(client, "MONTHLY_BUDGET_USD", 10.0)
    monkeypatch.setattr(client, "_month", lambda: "2026-10")
    client._record(9.5, "assistant")
    assert client.budget_left() == (9.5, 10.0)
    client._record(0.6, "assistant")
    assert client.budget_left()[0] >= 10.0                      # cap reached this month
    monkeypatch.setattr(client, "_month", lambda: "2026-11")
    assert client.budget_left() == (0.0, 10.0)                   # resets next month
    assert json.loads(ledger.read_text())["total_usd"] == pytest.approx(10.1)


def test_cumulative_cap_when_no_monthly_budget(ledger, monkeypatch):
    monkeypatch.setattr(client, "MONTHLY_BUDGET_USD", None)
    monkeypatch.setattr(client, "BUDGET_USD", 1.0)
    client._record(0.4, "eval")
    assert client.budget_left() == (0.4, 1.0)


def test_reached_cap_is_a_clean_503():
    from ensemble.api import app as app_module

    @app_module.app.get("/__test_budget")
    def boom():
        raise client.BudgetExceeded("LLM budget of $10 reached")

    r = TestClient(app_module.app).get("/__test_budget")
    assert r.status_code == 503 and r.json()["error"]["code"] == "ai_budget_reached"

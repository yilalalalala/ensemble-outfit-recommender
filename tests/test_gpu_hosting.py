"""The live site runs the local site's models on a Modal GPU (D-052). No GPU or network needed."""
import pytest

from ensemble.llm import client as llm


def test_hosted_backend_uses_the_local_models(monkeypatch):
    monkeypatch.setenv("ENSEMBLE_LLM", "ollama-modal")
    vision, assistant = llm.get_client("vision"), llm.get_client("assistant")
    assert isinstance(vision, llm.ModalOllamaClient) and isinstance(assistant, llm.ModalOllamaClient)
    assert (vision.model, assistant.model) == (llm.LOCAL_MODELS["vision"], llm.LOCAL_MODELS["assistant"])


def test_gpu_requests_stop_at_the_monthly_budget(monkeypatch):
    monkeypatch.setattr(llm, "gpu_month_spend", lambda: llm.GPU_MONTHLY_BUDGET_USD + 0.01)
    with pytest.raises(llm.BudgetExceeded):
        llm.ModalOllamaClient("qwen2.5vl:7b").chat("sys", [{"role": "user", "content": "hi"}])


def test_warming_is_a_no_op_locally(monkeypatch):
    monkeypatch.delenv("ENSEMBLE_LLM", raising=False)
    assert llm.warm_gpu() is False


def test_warm_endpoint_answers_locally():
    from fastapi.testclient import TestClient

    from ensemble.api.app import app
    r = TestClient(app).post("/api/stylist/warm")
    assert r.status_code == 200 and r.json() == {"warming": False}


def test_modal_app_serves_both_local_models_on_one_gpu_with_a_budget():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "deploy" / "modal_app.py").read_text()
    for m in llm.LOCAL_MODELS.values():
        assert f'"{m}"' in src, m
    assert '"ENSEMBLE_LLM": "ollama-modal"' in src and "GPU_MONTHLY_BUDGET_USD = 25" in src and "scaledown_window=300" in src

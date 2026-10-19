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
    assert 'GPU = ["L4", "A10G", "L40S"]' in src   # fallbacks when Modal has no L4 free


def test_a_gpu_that_does_not_answer_in_time_is_reported(monkeypatch):
    """No GPU capacity (or a slow cold start) must not hang the request: it raises GPUUnavailable after the wait."""
    import sys
    import types

    class Call:
        cancelled = False
        def get(self, timeout=None): raise TimeoutError()
        def cancel(self): Call.cancelled = True

    class Method:
        def spawn(self, body): return Call()

    class Obj:
        chat = Method()
    fake = types.SimpleNamespace(Cls=types.SimpleNamespace(from_name=lambda app, name: (lambda: Obj())),
                                 exception=types.SimpleNamespace(TimeoutError=TimeoutError, OutputExpiredError=TimeoutError))
    monkeypatch.setitem(sys.modules, "modal", fake)
    monkeypatch.setattr(llm, "gpu_month_spend", lambda: 0.0)
    with pytest.raises(llm.GPUUnavailable):
        llm.ModalOllamaClient("qwen2.5vl:7b")._post({})
    assert Call.cancelled


def test_detection_falls_back_to_claude_when_the_gpu_is_unavailable(monkeypatch):
    import io

    from fastapi.testclient import TestClient
    from PIL import Image

    import ensemble.vision.outfit as outfit
    from ensemble.api.app import app
    used = []

    def fake_detect(img, client, purpose="detect", **k):
        used.append(type(client).__name__)
        if type(client).__name__ == "ModalOllamaClient":
            raise llm.GPUUnavailable("no capacity")
        return [{"category": "dress", "colour": "white", "description": "dress", "box": [10, 10, 50, 90]}], img

    class FakeClaude: pass
    monkeypatch.setattr(outfit, "detect", fake_detect)
    monkeypatch.setattr(llm, "get_client", lambda role="assistant", backend=None, **k:
                        FakeClaude() if backend == "claude" else llm.ModalOllamaClient("m"))
    buf = io.BytesIO(); Image.new("RGB", (100, 100), "white").save(buf, format="JPEG")
    r = TestClient(app).post("/api/stylist/detect", files={"photo": ("p.jpg", buf.getvalue(), "image/jpeg")})
    assert r.status_code == 200 and r.json()["garments"][0]["category"] == "dress"
    assert used == ["ModalOllamaClient", "FakeClaude"]

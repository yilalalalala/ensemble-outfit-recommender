"""Low-cost production deployment for the real Ensemble application.

Deploy with ``modal deploy deploy/modal_app.py`` after uploading the serving
artifacts to the ``ensemble-serving-data`` Volume.  Catalogue photographs live
in Cloudflare R2 and are returned as direct URLs by the FastAPI application.
"""
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]


def requirements(name: str) -> list[str]:
    """Flatten a requirements file and its ``-r`` includes. Modal copies only the named file into the
    build, so nested includes must be resolved here."""
    out: list[str] = []
    for line in (ROOT / name).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        out += requirements(line[3:].strip()) if line.startswith("-r ") else [line]
    return out


# The image is defined on the deploying machine; inside the container this module is only imported.
_reqs = requirements("requirements-modal.txt") if modal.is_local() else []
# The container is CPU-only: install the CPU build of torch (far smaller image, faster cold start).
TORCH = [r for r in _reqs if r.startswith("torch")]
OTHER = [r for r in _reqs if not r.startswith("torch")]
# Public base URL of the catalogue photos in R2 (bucket prefix included), e.g. https://pub-<id>.r2.dev/catalog
IMAGE_BASE_URL = os.environ.get("ENSEMBLE_IMAGE_BASE_URL", "")

app = modal.App("ensemble-recommender")

# ---- the local site's models on a GPU (D-052) ------------------------------------------------------------------
# Ollama serves the two models the local site uses: Qwen2.5-VL 7B (garment detection) and Qwen3-VL 8B (the
# stylist's tool-calling chat). One L4 holds both. The container starts on demand (the site warms it when a shopper
# opens the AI Stylist), stays up 5 minutes after the last request, and bills its own uptime into a Modal Dict;
# past GPU_MONTHLY_BUDGET_USD the API refuses GPU requests and the stylist features rest until the next month.
OLLAMA_MODELS = ["qwen2.5vl:7b", "qwen3-vl:8b-instruct"]
# L4 first; when Modal has no L4 free (it happens), A10G, then L40S: all three hold both models. Billed per type.
GPU = ["L4", "A10G", "L40S"]
GPU_USD_PER_HOUR = {"L4": 0.80, "A10G": 1.10, "L40S": 1.95}
EXTRA_USD_PER_SECOND = 0.00003                # CPU and memory, rounded up
GPU_MONTHLY_BUDGET_USD = 25   # under the $30 free monthly credits with the web backend; Modal spend limit $5 behind it
models_volume = modal.Volume.from_name("ensemble-ollama-models", create_if_missing=True)
ollama_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "zstd")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .env({"OLLAMA_MODELS": "/models", "OLLAMA_HOST": "127.0.0.1:11434", "OLLAMA_MAX_LOADED_MODELS": "2",
          "OLLAMA_KEEP_ALIVE": "-1", "OLLAMA_NUM_PARALLEL": "2"})
)
data = modal.Volume.from_name("ensemble-serving-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*TORCH, index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(*OTHER)
    .env({
        "PYTHONPATH": "/root/src",
        "HF_HOME": "/root/.cache/huggingface",
        "TOKENIZERS_PARALLELISM": "false",
        "ENSEMBLE_PATH_REPORTS": "/root/data/reports",
        # The live site runs the same models as the local site: Ollama on a Modal GPU (OllamaGPU below, D-052).
        "ENSEMBLE_LLM": "ollama-modal",
        "ENSEMBLE_GPU_MONTHLY_BUDGET_USD": str(GPU_MONTHLY_BUDGET_USD),
        "ENSEMBLE_ASSISTANT_MODEL": "claude-haiku-4-5",
        "ENSEMBLE_VISION_MODEL": "claude-haiku-4-5",
        "ENSEMBLE_LLM_BUDGET_USD": "10",
        # Hard monthly cap on Claude spend (resets on the 1st, UTC); the counter lives in a Modal Dict so it
        # survives cold starts and redeploys. The Anthropic Console limit is a second, provider-side cap.
        "ENSEMBLE_LLM_MONTHLY_BUDGET_USD": "10",
        "ENSEMBLE_LLM_SPEND_DICT": "ensemble-llm-spend",
        "ENSEMBLE_LLM_SPEND_FILE": "/root/data/runtime/llm_spend.json",
        "ENSEMBLE_ALLOWED_ORIGINS": "https://yilalalalala.github.io,http://localhost:8010,http://localhost:8040",
        "ENSEMBLE_IMAGE_BASE_URL": IMAGE_BASE_URL,
        "ENSEMBLE_HOSTED": "1",   # disables the local-only Label tool (gold labels, private photo crops)
    })
    # Bake FashionCLIP into the image so the first public photo request does
    # not depend on a large Hugging Face download during cold start.
    .run_commands(
        "python -c \"from transformers import CLIPModel, CLIPProcessor; "
        "name='patrickjohncyh/fashion-clip'; "
        "CLIPModel.from_pretrained(name); CLIPProcessor.from_pretrained(name)\"",
        # The clothes parser that localises detected garments (vision/parsing.py, D-051), baked in for the same reason.
        "python -c \"from transformers import AutoModelForSemanticSegmentation; "
        "AutoModelForSemanticSegmentation.from_pretrained('mattmdjaga/segformer_b2_clothes')\"",
    )
    .add_local_dir(str(ROOT / "src"), remote_path="/root/src")
    .add_local_dir(str(ROOT / "configs"), remote_path="/root/configs")
)


@app.cls(image=ollama_image, gpu=GPU, volumes={"/models": models_volume}, timeout=900,
         scaledown_window=300, max_containers=1)
@modal.concurrent(max_inputs=4)
class OllamaGPU:
    @modal.enter()
    def start(self):
        import subprocess
        import time
        import urllib.request
        self.started = self.billed = time.time()
        name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout
        kind = next((k for k in GPU_USD_PER_HOUR if k in name.replace(" ", "")), "L40S")   # unknown: bill the dearest
        self.usd_per_second = GPU_USD_PER_HOUR[kind] / 3600 + EXTRA_USD_PER_SECOND
        print(f"[gpu start] running on {kind} ({name.strip()})", flush=True)
        lap = lambda what: print(f"[gpu start] {what}: {time.time() - self.started:.1f}s", flush=True)
        self.proc = subprocess.Popen(["ollama", "serve"])
        for _ in range(120):
            try:
                urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=2)
                break
            except Exception:  # noqa: BLE001 - the server is still starting
                time.sleep(0.5)
        lap("ollama serving")
        pulled = False
        for m in OLLAMA_MODELS:   # first start only: download into the volume
            if subprocess.run(["ollama", "show", m], capture_output=True).returncode != 0:
                subprocess.run(["ollama", "pull", m], check=True)
                pulled = True
        if pulled:
            models_volume.commit()
        for m in OLLAMA_MODELS:   # load both into GPU memory now, so the first request does not wait for it
            self._call({"model": m, "messages": [{"role": "user", "content": "hi"}], "stream": False,
                        "options": {"num_predict": 1}})
            lap(f"{m} loaded")
        self._bill()

    def _call(self, body: dict) -> dict:
        import json
        import urllib.request
        req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=600))

    def _bill(self):
        """Add the uptime since the last bill to this month's GPU spend (idle time included: it is billed too)."""
        import time
        now = time.time()
        d = modal.Dict.from_name("ensemble-gpu-spend", create_if_missing=True)
        key = time.strftime("%Y-%m", time.gmtime())
        d[key] = float(d.get(key, 0.0)) + (now - self.billed) * self.usd_per_second
        self.billed = now

    @modal.method()
    def chat(self, body: dict) -> dict:
        try:
            return self._call(body)
        finally:
            self._bill()

    @modal.method()
    def warm(self) -> bool:
        self._bill()
        return True

    @modal.exit()
    def stop(self):
        self._bill()
        self.proc.terminate()


@app.function(
    image=image,
    volumes={"/root/data": data},
    secrets=[modal.Secret.from_name("ensemble-production")],
    cpu=2.0,
    memory=16384,
    timeout=900,
    scaledown_window=60,
    max_containers=1,  # assistant sessions are intentionally process-local
)
@modal.concurrent(max_inputs=12)
@modal.asgi_app()
def web():
    from ensemble.api.app import app as fastapi_app

    return fastapi_app

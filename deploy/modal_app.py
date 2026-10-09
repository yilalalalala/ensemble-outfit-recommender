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
        "ENSEMBLE_LLM": "claude",
        "ENSEMBLE_ASSISTANT_MODEL": "claude-haiku-4-5",
        "ENSEMBLE_VISION_MODEL": "claude-haiku-4-5",
        "ENSEMBLE_LLM_BUDGET_USD": "10",
        "ENSEMBLE_LLM_SPEND_FILE": "/root/data/runtime/llm_spend.json",
        "ENSEMBLE_ALLOWED_ORIGINS": "https://yilalalalala.github.io,http://localhost:8010,http://localhost:8040",
        "ENSEMBLE_IMAGE_BASE_URL": IMAGE_BASE_URL,
    })
    # Bake FashionCLIP into the image so the first public photo request does
    # not depend on a large Hugging Face download during cold start.
    .run_commands(
        "python -c \"from transformers import CLIPModel, CLIPProcessor; "
        "name='patrickjohncyh/fashion-clip'; "
        "CLIPModel.from_pretrained(name); CLIPProcessor.from_pretrained(name)\""
    )
    .add_local_dir(str(ROOT / "src"), remote_path="/root/src")
    .add_local_dir(str(ROOT / "configs"), remote_path="/root/configs")
)


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

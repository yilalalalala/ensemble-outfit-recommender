# Track B serving API (production-oriented prototype; see docs/RUNBOOK_SERVING.md).
#
# The image contains code and pinned dependencies only. The serving bundle (built from the
# restricted H&M data by `make serving-bundle`) and, optionally, the product images are mounted
# at run time and are never baked into the image:
#
#   docker build -t ensemble-serving .
#   docker run --rm -p 8010:8010 \
#     -v "$PWD/data/processed/serving_bundles:/bundles:ro" \
#     -v "$PWD/data/raw/images:/app/data/raw/images:ro" ensemble-serving
#
# Build with --build-arg WITH_VISUAL=1 to add FashionCLIP visual search (CPU PyTorch, ~1 GB).
FROM python:3.11-slim

ARG WITH_VISUAL=0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    ENSEMBLE_BUNDLE=/bundles PYTHONPATH=/app/src

WORKDIR /app
COPY requirements-serve.txt requirements-serve-visual.txt ./
RUN if [ "$WITH_VISUAL" = "1" ]; then pip install -r requirements-serve-visual.txt; \
    else pip install -r requirements-serve.txt; fi
COPY pyproject.toml ./
COPY configs ./configs
COPY src ./src

RUN useradd --create-home --uid 10001 ensemble && mkdir -p /bundles && chown ensemble /app
USER ensemble
EXPOSE 8010
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8010/readyz').status == 200 else 1)"
CMD ["uvicorn", "ensemble.api.app:app", "--host", "0.0.0.0", "--port", "8010", "--workers", "1"]

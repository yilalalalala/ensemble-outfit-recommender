# Runbook: Track B serving API (Complete the Look, visual search, outfit photo)

A production-oriented **prototype**: one local process, no live traffic, no inventory feed,
no online A/B test. Design and measurements: `reports/TRACK_A_RESEARCH_TRACK_B_PRODUCTION_REPORT.md`,
ADRs D-041 and D-042.

## What runs where

| piece | where | notes |
| --- | --- | --- |
| Serving bundle | `data/processed/serving_bundles/<version>/` + `CURRENT` pointer | built offline from the restricted H&M data; never in git or in the image |
| API | `ensemble.api.app` (FastAPI, uvicorn, 1 worker) | imports neither LightGBM nor DuckDB; PyTorch only when a visual endpoint is first used |
| Product images | `data/raw/images/` (optional mount) | only for the UI cards; missing images render a placeholder |

## Build a bundle

```bash
make serving-bundle          # ~1 h; trains both rankers point-in-time, scores every live key,
                             # snapshots customer profiles, runs the equivalence gate, publishes
python -m ensemble.serving.bundle verify data/processed/serving_bundles/<version>   # full sha256 check
```

The build writes into `serving_bundles/.tmp-<ts>/`, renames it to its version only after the
**equivalence gate** passes (offline SQL + LightGBM vs online profile + NumPy runtime must give
identical orderings), then atomically replaces `CURRENT`. A failed build leaves `CURRENT` and the
running service untouched; the `.tmp-*` directory is kept for inspection.

## Start, check, stop

```bash
make serve-v2                               # http://localhost:8010, bundle = CURRENT
ENSEMBLE_BUNDLE=/path/to/bundle make serve-v2
curl -s localhost:8010/healthz              # liveness: process up
curl -s localhost:8010/readyz               # readiness: 200 only with a validated bundle loaded
curl -s localhost:8010/api/v2/meta          # model / catalogue / profile / availability versions
curl -s 'localhost:8010/api/v2/complete-the-look?anchor=<article_id>&customer=<customer_idx>&k=8'
curl -s localhost:8010/metrics              # Prometheus text; /api/v2/metrics is the JSON view
make serve-smoke                            # end-to-end check on a synthetic bundle (no data needed)
```

Container (code + pinned dependencies only; artifacts mounted read-only):

```bash
docker build -t ensemble-serving .                       # --build-arg WITH_VISUAL=1 adds FashionCLIP
docker run --rm -p 8010:8010 -v "$PWD/data/processed/serving_bundles:/bundles:ro" ensemble-serving
```

Environment: `ENSEMBLE_BUNDLE` (bundle dir or bundles root), `ENSEMBLE_CACHE_SIZE` (LRU entries,
0 disables), `ENSEMBLE_PERSONALIZATION=0` (kill switch: compatibility order for everyone),
`ENSEMBLE_VISUAL_TIMEOUT_S`, `ENSEMBLE_LOG_SALT` (customer-id pseudonymization in logs).

## Deploy a new bundle / roll back

```bash
# hot swap (local-only admin endpoint): the new bundle is fully loaded and validated first;
# on failure the old one keeps serving and the call returns 409 bundle_invalid
curl -s -X POST localhost:8010/admin/reload -H 'content-type: application/json' \
     -d '{"path": "data/processed/serving_bundles/<version>", "verify_hashes": true}'
# roll back = point CURRENT at the previous version, then reload (or restart)
python -m ensemble.serving.bundle publish <previous_version>
curl -s -X POST localhost:8010/admin/reload -H 'content-type: application/json' -d '{}'
```

Every cache key contains the bundle and availability versions and a swap starts a fresh cache,
so no recommendation from the old bundle is served after a swap.

## Stock-outs (no inventory feed exists)

```bash
curl -s -X POST localhost:8010/admin/availability -H 'content-type: application/json' \
     -d '{"unavailable": [706016001, 372860002]}'      # replaces the override list
```

Unavailable articles are filtered **before** the top-k cut; modules are back-filled from slot
popularity, and the response's `availability_version` changes.

## Common failures

| symptom | cause | action |
| --- | --- | --- |
| `/readyz` 503 `no serving bundle at …` | no bundle built / wrong `ENSEMBLE_BUNDLE` | `make serving-bundle`, or point `ENSEMBLE_BUNDLE` at a bundle dir |
| `/readyz` 503 `missing [...]` / `wrong size` / `sha256` | partial copy or corruption | rebuild, or roll back to the previous version |
| `/readyz` 503 `feature schema hash` / `features differ` | model and pools from different builds | never mix files across versions; rebuild |
| visual endpoints 503 `visual_model_unavailable` | FashionCLIP weights not cached / no visual index | Complete the Look by id still works; install `requirements-serve-visual.txt`, warm the HF cache |
| visual responses `mode: "crop"`, `degraded: true` | DeepFashion2 adapter absent | expected degraded mode (D-021 runner-up) |
| outfit 503 `detector_unavailable` | local VLM (Ollama) down or slow; paid backends are refused | send the user-selected `garments` JSON instead |
| 413 / 415 / 400 / 422 on uploads | size, type, decode or dimension limits (`/api/v2/meta` → `upload_limits`) | expected; nothing is stored or logged |
| p99 latency spikes after a swap | cold cache | expected; see the warm vs cold benchmark |

## Privacy

Uploads live in memory for one request and are never written to disk or logged. Request logs
carry a salted hash of the customer id, never the id, the history or image bytes.

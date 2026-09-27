"""MLflow experiment tracking: SQLite backend store and local artifacts under mlruns/.

Browse with: mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
"""
from __future__ import annotations

import os
import re
from contextlib import contextmanager

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow  # noqa: E402

from ensemble.config import load_config  # noqa: E402


@contextmanager
def run(experiment: str, name: str, params: dict | None = None):
    root = load_config().path("mlruns")
    root.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(root / "artifacts" / experiment).as_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=name) as r:
        if params:
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        yield r


def log_metrics(metrics: dict, prefix: str = "") -> None:
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and v == v:  # drop NaN
            name = f"{prefix}{k}".replace("@", "_at_").replace("+", "_plus_")
            clean[re.sub(r"[^\w\-. :/]", "_", name)] = float(v)
    mlflow.log_metrics(clean)

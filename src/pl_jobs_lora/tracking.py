"""MLflow tracking wiring — one place that decides where runs are logged (ADR-0004).

The hosted MLflow endpoint (DagsHub, for parity with P1 mlops-car-price) is supplied at
runtime via the ``MLFLOW_TRACKING_URI`` env var, which wins over the config value. That way
the same code logs to a local file store during development and to the hosted tracker from
Colab, with no code change. ``mlflow`` is imported lazily so the rest of the package (schema,
normalize, scorer) stays importable without the tracking client.
"""

from __future__ import annotations

import os

from pl_jobs_lora.config import Config, load_config

_ENV_TRACKING_URI = "MLFLOW_TRACKING_URI"


def resolve_tracking_uri(config: Config | None = None) -> str | None:
    """Return the tracking URI: the env var wins, then config, then None (local store)."""
    env = os.environ.get(_ENV_TRACKING_URI)
    if env:
        return env
    config = config or load_config()
    return config.tracking.tracking_uri


def start_run(run_name: str, config: Config | None = None):
    """Configure MLflow from the resolved URI + experiment and open a run (lazy import)."""
    import mlflow

    config = config or load_config()
    uri = resolve_tracking_uri(config)
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(config.tracking.experiment)
    return mlflow.start_run(run_name=run_name)

"""Config loads into frozen dataclasses; the tracking URI honors the env override."""

from dataclasses import FrozenInstanceError

import pytest

from pl_jobs_lora.config import load_config
from pl_jobs_lora.tracking import resolve_tracking_uri


def test_config_loads_candidates_and_probe():
    cfg = load_config()
    assert len(cfg.models) >= 2  # the probe needs a comparison
    keys = {m.key for m in cfg.models}
    assert {"qwen2.5-1.5b", "bielik-1.5b"} <= keys
    assert cfg.probe.temperature == 0.0  # greedy — base model is the only variable
    assert cfg.probe.dev_slice_size > 0


def test_config_is_frozen():
    cfg = load_config()
    with pytest.raises(FrozenInstanceError):
        cfg.probe.temperature = 0.9  # type: ignore[misc]


def test_tracking_uri_env_wins(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://example.test/mlflow")
    assert resolve_tracking_uri() == "https://example.test/mlflow"


def test_tracking_uri_falls_back_to_config(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    # config ships tracking_uri: null → local store
    assert resolve_tracking_uri() is None

"""Configuration: YAML → frozen dataclasses. No I/O beyond reading the config file.

Mirrors the P1 mlops-car-price pattern — every knob (model candidates, probe decoding,
dataset size, HF repos, tracking) lives in configs/config.yaml, loaded once into immutable
dataclasses so nothing is hardcoded and invalid states are caught at load time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


@dataclass(frozen=True)
class ModelCandidate:
    key: str
    hf_repo: str
    gguf_repo: str | None
    gguf_file: str | None


@dataclass(frozen=True)
class ProbeConfig:
    dev_slice_size: int
    temperature: float
    max_tokens: int
    few_shot_examples: int
    context_tokens: int
    shot_modes: tuple[str, ...]


@dataclass(frozen=True)
class CollectionConfig:
    sitemap_url: str
    user_agent: str
    request_delay_s: float
    request_timeout_s: int


@dataclass(frozen=True)
class ScoringConfig:
    salary_rel_tolerance: float


@dataclass(frozen=True)
class DataConfig:
    sitemap_offers_sample: int
    hf_dataset_repo: str


@dataclass(frozen=True)
class HfConfig:
    adapter_repo: str


@dataclass(frozen=True)
class TrackingConfig:
    experiment: str
    tracking_uri: str | None


@dataclass(frozen=True)
class Config:
    models: tuple[ModelCandidate, ...]
    probe: ProbeConfig
    data: DataConfig
    hf: HfConfig
    tracking: TrackingConfig
    collection: CollectionConfig
    scoring: ScoringConfig


def load_config(path: Path | None = None) -> Config:
    """Load and validate the config into an immutable ``Config``."""
    path = path or _DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    candidates = tuple(
        ModelCandidate(
            key=c["key"], hf_repo=c["hf_repo"],
            gguf_repo=c.get("gguf_repo"), gguf_file=c.get("gguf_file"),
        )
        for c in raw["models"]["candidates"]
    )
    if len(candidates) < 2:
        raise ValueError("The base-model probe needs at least two candidates (ADR-0001).")
    probe_raw = dict(raw["probe"])
    probe_raw["shot_modes"] = tuple(probe_raw["shot_modes"])
    return Config(
        models=candidates,
        probe=ProbeConfig(**probe_raw),
        data=DataConfig(**raw["data"]),
        hf=HfConfig(**raw["hf"]),
        tracking=TrackingConfig(**raw["tracking"]),
        collection=CollectionConfig(**raw["collection"]),
        scoring=ScoringConfig(**raw["scoring"]),
    )

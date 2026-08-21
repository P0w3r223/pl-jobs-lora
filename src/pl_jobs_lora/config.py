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
    context_margin_tokens: int
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
    bootstrap_resamples: int
    bootstrap_seed: int
    bootstrap_ci: float


@dataclass(frozen=True)
class EvalConfig:
    api_model: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    max_tokens: int
    temperature: float
    few_shot_examples: int
    shot_modes: tuple[str, ...]
    request_max_snippets: int
    latency_percentiles: tuple[int, ...]


@dataclass(frozen=True)
class LabelingQaConfig:
    proposer_backend: str
    proposer: str
    proposer_mode: str
    arbiter: str
    arbiter_max_snippets: int
    human_sample_size: int
    sampling: str
    sampling_seed: int
    validated_f1: float | None


@dataclass(frozen=True)
class TrainConfig:
    base: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: str
    bnb_4bit_quant_type: str
    bnb_4bit_use_double_quant: bool
    compute_dtype: str
    epochs: int
    learning_rate: float
    lr_scheduler: str
    warmup_ratio: float
    per_device_batch_size: int
    grad_accum_steps: int
    max_seq_len: int
    dev_fraction: float
    seed: int


@dataclass(frozen=True)
class DataConfig:
    sitemap_offers_sample: int
    test_fraction: float
    min_prose_chars: int
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
    eval: EvalConfig
    train: TrainConfig
    labeling_qa: LabelingQaConfig


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
    eval_raw = dict(raw["eval"])
    eval_raw["shot_modes"] = tuple(eval_raw["shot_modes"])
    eval_raw["latency_percentiles"] = tuple(eval_raw["latency_percentiles"])
    eval_cfg = EvalConfig(**eval_raw)
    _validate_eval(eval_cfg)
    train_cfg = TrainConfig(**raw["train"])
    _validate_train(train_cfg, candidates)
    labeling_qa = LabelingQaConfig(**raw["labeling_qa"])
    _validate_labeling_qa(labeling_qa, candidates)
    return Config(
        models=candidates,
        probe=ProbeConfig(**probe_raw),
        data=DataConfig(**raw["data"]),
        hf=HfConfig(**raw["hf"]),
        tracking=TrackingConfig(**raw["tracking"]),
        collection=CollectionConfig(**raw["collection"]),
        scoring=ScoringConfig(**raw["scoring"]),
        eval=eval_cfg,
        train=train_cfg,
        labeling_qa=labeling_qa,
    )


def _validate_train(cfg: TrainConfig, candidates: tuple[ModelCandidate, ...]) -> None:
    """Fail fast on QLoRA knobs (ADR-0006) so a bad config never reaches the GPU trainer."""
    if cfg.base not in {c.key for c in candidates}:
        raise ValueError(f"train.base={cfg.base!r} is not a known models.candidates key.")
    if cfg.lora_r <= 0 or cfg.lora_alpha <= 0:
        raise ValueError("train.lora_r and train.lora_alpha must be positive.")
    if not 0.0 <= cfg.lora_dropout < 1.0:
        raise ValueError("train.lora_dropout must be within [0, 1).")
    if cfg.compute_dtype not in {"fp16", "bf16"}:
        raise ValueError(f"train.compute_dtype must be 'fp16'/'bf16', got {cfg.compute_dtype!r}.")
    if not 0.0 <= cfg.warmup_ratio < 1.0:
        raise ValueError("train.warmup_ratio must be within [0, 1).")
    if cfg.epochs <= 0:
        raise ValueError("train.epochs must be positive.")
    if cfg.learning_rate <= 0:
        raise ValueError("train.learning_rate must be positive.")
    if cfg.per_device_batch_size <= 0 or cfg.grad_accum_steps <= 0:
        raise ValueError("train.per_device_batch_size and train.grad_accum_steps must be positive.")
    if cfg.max_seq_len <= 0:
        raise ValueError("train.max_seq_len must be positive.")
    if not 0.0 < cfg.dev_fraction < 1.0:
        raise ValueError("train.dev_fraction must be within (0, 1).")


def _validate_eval(cfg: EvalConfig) -> None:
    """Fail fast on eval knobs (ADR-0003) so a bad config never reaches the baseline CLI."""
    unknown = set(cfg.shot_modes) - {"zero", "few"}
    if unknown:
        raise ValueError(f"eval.shot_modes must be a subset of {{zero, few}}, got {unknown}.")
    if cfg.few_shot_examples < 0:
        raise ValueError("eval.few_shot_examples must be non-negative.")
    if cfg.input_usd_per_mtok <= 0 or cfg.output_usd_per_mtok <= 0:
        raise ValueError("eval.{input,output}_usd_per_mtok must be positive (pull list price).")
    if cfg.request_max_snippets <= 0:
        raise ValueError("eval.request_max_snippets must be positive.")
    if any(not 0 <= p <= 100 for p in cfg.latency_percentiles):
        raise ValueError("eval.latency_percentiles must each be within [0, 100].")


def _validate_labeling_qa(
    cfg: LabelingQaConfig, candidates: tuple[ModelCandidate, ...]
) -> None:
    """Fail fast on labeling-QA knobs (ADR-0005) so a bad config never reaches the CLI."""
    if cfg.proposer_backend != "gguf":
        raise ValueError(
            f"labeling_qa.proposer_backend={cfg.proposer_backend!r} is not implemented "
            "(YAGNI); only 'gguf' is supported in S3 (ADR-0005)."
        )
    if cfg.proposer not in {c.key for c in candidates}:
        raise ValueError(
            f"labeling_qa.proposer={cfg.proposer!r} is not a known models.candidates key."
        )
    if cfg.proposer_mode not in {"zero", "few"}:
        raise ValueError(
            f"labeling_qa.proposer_mode must be 'zero' or 'few', got {cfg.proposer_mode!r}."
        )
    if cfg.sampling not in {"random", "stratified"}:
        raise ValueError(
            f"labeling_qa.sampling must be 'random' or 'stratified', got {cfg.sampling!r}."
        )
    if cfg.human_sample_size <= 0:
        raise ValueError("labeling_qa.human_sample_size must be positive.")
    if cfg.arbiter_max_snippets <= 0:
        raise ValueError("labeling_qa.arbiter_max_snippets must be positive.")

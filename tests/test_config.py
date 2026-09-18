"""Config loads into frozen dataclasses; the tracking URI honors the env override."""

import dataclasses
from dataclasses import FrozenInstanceError

import pytest

from pl_jobs_lora.config import (
    DataConfig,
    LabelingQaConfig,
    ProbeConfig,
    TrainConfig,
    _validate_data,
    _validate_labeling_qa,
    _validate_probe,
    _validate_train,
    load_config,
)
from pl_jobs_lora.tracking import resolve_tracking_uri


def test_config_loads_candidates_and_probe():
    cfg = load_config()
    assert len(cfg.models) >= 2  # the probe needs a comparison
    keys = {m.key for m in cfg.models}
    assert {"qwen2.5-1.5b", "bielik-1.5b"} <= keys
    assert cfg.probe.temperature == 0.0  # greedy — base model is the only variable
    assert cfg.probe.dev_slice_size > 0


def _valid_probe(**overrides) -> ProbeConfig:
    base = dict(
        dev_slice_size=25, temperature=0.0, max_tokens=2048, few_shot_examples=2,
        context_tokens=8192, context_margin_tokens=128, shot_modes=("zero", "few"),
    )
    return ProbeConfig(**{**base, **overrides})


def _valid_data(**overrides) -> DataConfig:
    base = dict(
        sitemap_offers_sample=800, test_fraction=0.2, min_prose_chars=200,
        hf_dataset_repo="example/dataset",
    )
    return DataConfig(**{**base, **overrides})


def test_the_two_sample_sizes_that_become_a_divisor_are_refused_at_load():
    """`_spread_sample` divides by both, so a zero here used to crash inside the collector.

    `probe.dev_slice_size` and `data.sitemap_offers_sample` are the two knobs that reach
    `step = max(1, len(urls) // n)`. Before these validators a `0` in `configs/config.yaml`
    surfaced as `ZeroDivisionError` from the network half of `dataset/collect.py`, which is a
    configuration error arriving as a crash three modules away — `good-practices.md` §2 asks
    the boundary to refuse it instead.
    """
    with pytest.raises(ValueError, match="dev_slice_size"):
        _validate_probe(_valid_probe(dev_slice_size=0))
    with pytest.raises(ValueError, match="sitemap_offers_sample"):
        _validate_data(_valid_data(sitemap_offers_sample=0))


def test_the_probe_validator_holds_the_rest_of_its_block():
    for bad in (dict(temperature=-0.1), dict(max_tokens=0), dict(few_shot_examples=-1),
                dict(context_tokens=0), dict(context_margin_tokens=8192),
                dict(shot_modes=("zero", "many"))):
        with pytest.raises(ValueError):
            _validate_probe(_valid_probe(**bad))
    _validate_probe(_valid_probe())  # the positive control: the shipped shape passes


def test_the_data_validator_holds_the_rest_of_its_block():
    for bad in (dict(test_fraction=0.0), dict(test_fraction=1.0), dict(min_prose_chars=0)):
        with pytest.raises(ValueError):
            _validate_data(_valid_data(**bad))
    _validate_data(_valid_data())


def test_the_shipped_config_passes_both_new_validators():
    """Load is the assertion: `load_config` calls them, so this fails if the file drifts."""
    cfg = load_config()
    assert cfg.probe.dev_slice_size > 0
    assert cfg.data.sitemap_offers_sample > 0


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


def test_labeling_qa_block_loads(monkeypatch):
    cfg = load_config()
    lq = cfg.labeling_qa
    assert lq.proposer_backend == "gguf"  # api is the deferred seam (ADR-0005)
    assert lq.proposer in {m.key for m in cfg.models}  # proposer references a real candidate
    assert lq.proposer != lq.arbiter  # arbiter distinct → LLM<->human isn't self-agreement
    assert 60 <= lq.human_sample_size <= 80
    assert lq.arbiter_max_snippets > 0
    assert lq.validated_f1 is None  # soft flag off by default, never a gate


def _valid_lq(**overrides) -> LabelingQaConfig:
    base = dict(
        proposer_backend="gguf", proposer="bielik-1.5b", proposer_mode="few",
        arbiter="claude-opus-4-8", arbiter_max_snippets=80, human_sample_size=72,
        sampling="stratified", sampling_seed=20260728, validated_f1=None,
    )
    return LabelingQaConfig(**{**base, **overrides})


@pytest.mark.parametrize(
    "overrides",
    [
        {"proposer_backend": "api"},        # deferred seam must fail fast (YAGNI)
        {"proposer": "gpt-nonexistent"},    # unknown candidate key
        {"proposer_mode": "many"},          # not zero|few
        {"sampling": "grid"},               # not random|stratified
        {"human_sample_size": 0},           # must be positive
        {"arbiter_max_snippets": 0},        # egress cap must be positive
    ],
)
def test_labeling_qa_validation_rejects_bad_config(overrides):
    cfg = load_config()
    with pytest.raises(ValueError):
        _validate_labeling_qa(_valid_lq(**overrides), cfg.models)


def test_labeling_qa_config_is_frozen():
    lq = _valid_lq()
    with pytest.raises(FrozenInstanceError):
        lq.human_sample_size = 5  # type: ignore[misc]
    # sanity: the helper builds a fully-populated dataclass (no missing fields)
    assert len(dataclasses.fields(LabelingQaConfig)) == len(dataclasses.asdict(lq))


def test_train_block_loads():
    t = load_config().train
    assert t.base in {"qwen2.5-1.5b", "bielik-1.5b"}  # references a real candidate (ADR-0006)
    assert t.target_modules == "all-linear" and t.lora_r > 0
    assert t.compute_dtype in {"fp16", "bf16"}
    assert 0.0 < t.dev_fraction < 1.0


def _valid_train(**overrides) -> TrainConfig:
    base = dict(
        base="bielik-1.5b", lora_r=16, lora_alpha=32, lora_dropout=0.1, target_modules="all-linear",
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, compute_dtype="fp16",
        epochs=3, learning_rate=2e-4, lr_scheduler="cosine", warmup_ratio=0.05,
        per_device_batch_size=4, grad_accum_steps=4, max_seq_len=2048, dev_fraction=0.1,
        seed=20260728,
    )
    return TrainConfig(**{**base, **overrides})


@pytest.mark.parametrize(
    "overrides",
    [
        {"base": "gpt-nonexistent"},      # unknown candidate key
        {"lora_r": 0},                    # must be positive
        {"lora_dropout": 1.0},            # must be within [0, 1)
        {"compute_dtype": "int8"},        # not fp16|bf16
        {"warmup_ratio": 1.5},            # must be within [0, 1)
        {"epochs": 0},                    # must be positive
        {"dev_fraction": 0.0},            # must be within (0, 1)
    ],
)
def test_train_validation_rejects_bad_config(overrides):
    cfg = load_config()
    with pytest.raises(ValueError):
        _validate_train(_valid_train(**overrides), cfg.models)

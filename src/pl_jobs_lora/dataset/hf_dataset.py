"""Freeze the processed dataset locally and (later) push it to the HF Hub.

ADR-0002 freezes the train/test JSONL on the Hub so every downstream step replays one immutable
dataset instead of re-scraping expiring offers. This module writes those files plus a manifest;
``push_dataset`` uploads them. The push is intentionally *not* run during S2 — it needs an HF
token — so the code is ready but gated behind an explicit call, mirroring the deferred GitHub
publish. Raw HTML never appears here: only prose-derived fields were ever kept (ADR-0002).
"""

from __future__ import annotations

import json
from pathlib import Path

from pl_jobs_lora.config import Config


def write_jsonl(records: list[dict], path: Path) -> None:
    """Write records one-JSON-per-line (UTF-8, Polish text preserved)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _date_range(records: list[dict]) -> dict | None:
    dates = sorted(r["pub_date"] for r in records if r.get("pub_date"))
    return {"earliest": dates[0], "latest": dates[-1]} if dates else None


def dataset_manifest(
    train: list[dict], test: list[dict], stats: dict, cfg: Config
) -> dict:
    """Pure provenance record for the frozen dataset: counts, date ranges, and the knobs that
    produced it. Frozen alongside the JSONL so a reader can trust and reproduce the split."""
    return {
        "hf_dataset_repo": cfg.data.hf_dataset_repo,
        "splits": {
            "train": {"n": len(train), "dates": _date_range(train)},
            "test": {"n": len(test), "dates": _date_range(test)},
        },
        "build": stats,
        "params": {
            "sitemap_offers_sample": cfg.data.sitemap_offers_sample,
            "test_fraction": cfg.data.test_fraction,
            "min_prose_chars": cfg.data.min_prose_chars,
        },
        "split_strategy": "temporal (newest test_fraction -> test); never random (ADR-0002)",
    }


def push_dataset(cfg: Config, data_dir: Path, token: str | None = None) -> str:
    """Upload the frozen ``data_dir`` (train/test JSONL + manifest) to the HF Hub dataset repo.

    DEFERRED for S2: requires an HF write token (arg or ``HF_TOKEN`` env). Kept ready so the
    push is one call once the token is available — no re-collection needed."""
    import os

    from huggingface_hub import HfApi

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("No HF token: pass token= or set HF_TOKEN (dataset push is deferred).")
    api = HfApi(token=token)
    repo_id = cfg.data.hf_dataset_repo
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(data_dir), repo_id=repo_id, repo_type="dataset")
    return repo_id


def pull_dataset(cfg: Config, data_dir: Path, token: str | None = None) -> Path:
    """Download the frozen train/test JSONL + manifest from the HF Hub into ``data_dir``.

    The mirror of ``push_dataset``: the hosted GPU (and any fresh clone) fetches the immutable
    dataset instead of re-scraping, since ``data/processed/`` is gitignored. The dataset repo is
    private, so a read token is needed (arg, ``HF_TOKEN`` env, or a cached
    ``huggingface-cli login``)."""
    import os

    from huggingface_hub import hf_hub_download

    token = token or os.environ.get("HF_TOKEN")
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "test.jsonl", "manifest.json"):
        hf_hub_download(
            cfg.data.hf_dataset_repo, name, repo_type="dataset",
            local_dir=str(data_dir), token=token,
        )
    return data_dir

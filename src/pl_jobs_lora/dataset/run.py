"""Build the S2 dataset: collect -> build records -> temporal split -> freeze locally.

Orchestration and I/O only; the logic lives in the pure ``collect``/``build``/``split`` modules.
Collection is cached to a gitignored slice so the build can be re-run offline (replay, ADR-0002)
without re-fetching. The HF push is deferred (``--push`` gated on an HF token).

    python -m pl_jobs_lora.dataset.run --collect --limit 20   # smoke: fetch 20, build, split
    python -m pl_jobs_lora.dataset.run --collect              # full ~800 collection + freeze
    python -m pl_jobs_lora.dataset.run                        # replay cached slice, rebuild
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from pl_jobs_lora.config import Config, load_config
from pl_jobs_lora.dataset import build, hf_dataset, split
from pl_jobs_lora.dataset.collect import DevExample, collect_dataset
from pl_jobs_lora.normalize import load_tech_aliases

_ROOT = Path(__file__).resolve().parents[3]
_SLICE = _ROOT / "data" / "dataset_slice.json"      # gitignored: prose-derived, replayed
_PROCESSED = _ROOT / "data" / "processed"           # gitignored: frozen -> HF, not git


def save_slice(examples: list[DevExample], path: Path = _SLICE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(e) for e in examples], ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_slice(path: Path = _SLICE) -> list[DevExample]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [DevExample(**r) for r in rows]


def build_and_freeze(cfg: Config, examples: list[DevExample], out_dir: Path) -> dict:
    """Records -> temporal split -> train/test JSONL + manifest under ``out_dir`` (returns it)."""
    records, stats = build.build_dataset(examples, min_prose_chars=cfg.data.min_prose_chars)
    train, test = split.temporal_split(records, test_fraction=cfg.data.test_fraction)
    hf_dataset.write_jsonl(train, out_dir / "train.jsonl")
    hf_dataset.write_jsonl(test, out_dir / "test.jsonl")
    manifest = hf_dataset.dataset_manifest(train, test, stats, cfg)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the S2 prose->JSON dataset (ADR-0002).")
    ap.add_argument("--collect", action="store_true", help="fetch a fresh slice (else replay)")
    ap.add_argument("--limit", type=int, default=0, help="cap collection for a smoke run (0=full)")
    ap.add_argument("--push", action="store_true", help="upload to HF Hub (needs HF_TOKEN)")
    args = ap.parse_args()

    cfg = load_config()
    if args.collect:
        examples = collect_dataset(cfg, load_tech_aliases(), limit=args.limit)
        save_slice(examples)
    else:
        examples = load_slice()
    print(f"[dataset] {len(examples)} collected examples")

    manifest = build_and_freeze(cfg, examples, _PROCESSED)
    s = manifest["splits"]
    print(f"[dataset] records={manifest['build']['records']} "
          f"(dropped {manifest['build']['dropped_filters']} filtered, "
          f"{manifest['build']['dropped_duplicates']} dup)")
    print(f"[dataset] train={s['train']['n']} test={s['test']['n']} -> {_PROCESSED}")

    if args.push:
        repo = hf_dataset.push_dataset(cfg, _PROCESSED)
        print(f"[dataset] pushed to HF: {repo}")


if __name__ == "__main__":
    main()

"""Optional CPU-latency variant of the base (S5; ADR-0006).

Reuses the ADR-0001 probe machinery (``probe.run_inference``, whose llama-cpp import is lazy) to
time the untuned base on **local CPU via GGUF** over the frozen test set, producing a
``{base}-gguf__{mode}`` prediction file. This exists for the latency column of the comparison — the
CPU cost story ("a 1.5B model runs on a laptop") — not to re-rank accuracy; the adapter is not
GGUF-converted. Needs the optional ``gguf`` extra (llama-cpp-python).

    python -m pl_jobs_lora.inference.predict_gguf            # few-shot base on CPU
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pl_jobs_lora.config import Config, load_config
from pl_jobs_lora.eval.baselines import to_dev_examples, write_predictions
from pl_jobs_lora.train.qlora import resolve_base

_ROOT = Path(__file__).resolve().parents[3]
_PROCESSED = _ROOT / "data" / "processed"
_PRED_DIR = _ROOT / "results" / "eval" / "predictions"


def _read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def run_gguf_predictions(
    cfg: Config, *, mode: str = "few", processed_dir: Path = _PROCESSED,
    pred_dir: Path = _PRED_DIR, limit: int = 0,
) -> tuple[str, list[dict]]:
    """Time the base GGUF on the test set → ``{base}-gguf__{mode}`` preds. Needs llama-cpp."""
    from pl_jobs_lora import probe  # local: probe.run_inference lazily imports llama-cpp

    base = resolve_base(cfg)
    test = _read_jsonl(processed_dir / "test.jsonl")
    train = _read_jsonl(processed_dir / "train.jsonl")
    shots = to_dev_examples(train[: cfg.probe.few_shot_examples])
    eval_set = to_dev_examples(test[:limit] if limit else test)

    preds = probe.run_inference(base, mode, eval_set, shots, cfg)
    variant = f"{base.key}-gguf__{mode}"
    write_predictions(variant, preds, pred_dir)
    return variant, preds


def main() -> None:
    ap = argparse.ArgumentParser(description="CPU-latency GGUF base inference over the test set.")
    ap.add_argument("--mode", choices=["zero", "few"], default="few", help="shot mode")
    ap.add_argument("--limit", type=int, default=0, help="cap test examples for a smoke run")
    args = ap.parse_args()

    cfg = load_config()
    variant, preds = run_gguf_predictions(cfg, mode=args.mode, limit=args.limit)
    valid = sum(int(p["valid"]) for p in preds)
    print(f"[predict-gguf] {variant}: {len(preds)} predictions, {valid} valid JSON")


if __name__ == "__main__":
    main()

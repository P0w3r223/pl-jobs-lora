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
import time
from pathlib import Path

from pl_jobs_lora import resume
from pl_jobs_lora.config import Config, load_config
from pl_jobs_lora.eval.baselines import to_dev_examples
from pl_jobs_lora.train.qlora import resolve_base

_ROOT = Path(__file__).resolve().parents[3]
_PROCESSED = _ROOT / "data" / "processed"
_PRED_DIR = _ROOT / "results" / "eval" / "predictions"


def _read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# A current prediction row carries the failure taxonomy (ADR-0003). Rows without it were written
# before the taxonomy existed: they are re-run rather than resumed, which is how the taxonomy gets
# backfilled onto variants that were measured earlier.
_CURRENT_ROW_KEYS = ("failure", "raw")


def _decoded_under(cap: int):
    """A cached row counts as done only if it decoded under the cap this run uses.

    The cap is not bookkeeping — it is part of what was measured. A row cut off at 1024 tokens
    would have kept going under 2048, so keeping it would leave a file that averages latency and
    counts truncations across two decoding budgets while presenting one variant. Rows that predate
    the ``max_tokens`` field carry no cap at all and cannot prove which one they ran under, so they
    are outstanding too.
    """
    return lambda row: row.get("max_tokens") == cap


def run_gguf_predictions(
    cfg: Config, *, mode: str = "few", processed_dir: Path = _PROCESSED,
    pred_dir: Path = _PRED_DIR, limit: int = 0, fresh: bool = False,
    on_progress=None,
) -> tuple[str, list[dict]]:
    """Time the base GGUF on the test set → ``{base}-gguf__{mode}`` preds. Needs llama-cpp.

    **Resumable.** A full pass is ~50 min few-shot and ~2 h 40 min zero-shot on CPU, so the run is
    interruptible: each prediction is appended as it is produced, and a restart skips whatever is
    already on disk. Stop it with Ctrl-C and run the same command later to finish.

    Rows written before the failure taxonomy existed do not count as done, and neither do rows that
    decoded under a different ``probe.max_tokens``: re-running them is what backfills
    `failure`/`raw` and what keeps one prediction file from averaging two decoding budgets. Because
    that discards the old rows, the previous file is copied aside first under a suffix naming the
    configuration it held — they are regenerable, but regenerating them costs the hours this path
    exists to protect. ``fresh=True`` forces every record to be recomputed.
    """
    from pl_jobs_lora import probe  # local: probe.run_inference lazily imports llama-cpp

    base = resolve_base(cfg)
    test = _read_jsonl(processed_dir / "test.jsonl")
    train = _read_jsonl(processed_dir / "train.jsonl")
    # shots + decoding come from the probe config (this reuses probe.run_inference wholesale); it is
    # latency-only, so it need not match eval.few_shot_examples (same value today anyway).
    shots = to_dev_examples(train[: cfg.probe.few_shot_examples])
    eval_recs = test[:limit] if limit else test
    eval_set = to_dev_examples(eval_recs)

    variant = f"{base.key}-gguf__{mode}"
    path = pred_dir / f"{variant}.jsonl"
    done = {} if fresh else resume.load_completed(
        path, required_keys=_CURRENT_ROW_KEYS, matches=_decoded_under(cfg.probe.max_tokens),
    )
    todo = [ex for ex in eval_set if ex.offer_id not in done]

    if todo:
        # About to drop rows this run cannot use — keep a copy of what they measured.
        saved = resume.backup_superseded(path, done, required_keys=_CURRENT_ROW_KEYS)
        if saved is not None:
            print(f"[predict-gguf] superseded rows saved to {saved.name}")
        print(f"[predict-gguf] {variant}: {len(done)} cached, {len(todo)} to run")
        with resume.append_sink(path, done) as sink:
            def _sink(pred: dict) -> None:
                sink(pred)
                if on_progress is not None:
                    on_progress(len(done), len(eval_set))

            probe.run_inference(base, mode, todo, shots, cfg, on_prediction=_sink)

    # Deterministic file content once complete: the append order of a resumed run is an artefact of
    # when it was interrupted, not of the data.
    preds = [done[ex.offer_id] for ex in eval_set if ex.offer_id in done]
    if len(preds) == len(eval_set):
        resume.write_jsonl(path, preds)
    return variant, preds


def _progress_reporter():
    """Print a one-line ETA per record — a multi-hour CPU run must be observable while it runs.

    The rate comes from the records *this* process produced, not from the running total: on a
    resume the total includes work done by an earlier run, and dividing this run's elapsed time by
    it reports a speed the machine never achieved. Progress is still shown against the whole set,
    because 53/142 is what the operator wants to see, not 1/90.
    """
    started = time.perf_counter()
    produced = 0

    def report(done: int, total: int) -> None:
        nonlocal produced
        produced += 1
        elapsed = time.perf_counter() - started
        remaining = (elapsed / produced) * (total - done)
        print(
            f"[predict-gguf] {done}/{total}  elapsed {elapsed / 60:.1f} min  "
            f"eta {remaining / 60:.1f} min",
            flush=True,
        )

    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description="CPU-latency GGUF base inference over the test set (resumable).",
    )
    ap.add_argument("--mode", choices=["zero", "few", "both"], default="few", help="shot mode")
    ap.add_argument("--limit", type=int, default=0, help="cap test examples for a smoke run")
    ap.add_argument(
        "--fresh", action="store_true",
        help="recompute every record instead of resuming from the predictions file",
    )
    args = ap.parse_args()

    cfg = load_config()
    modes = ["few", "zero"] if args.mode == "both" else [args.mode]
    for mode in modes:
        variant, preds = run_gguf_predictions(
            cfg, mode=mode, limit=args.limit, fresh=args.fresh,
            on_progress=_progress_reporter(),
        )
        valid = sum(int(p["valid"]) for p in preds)
        print(f"[predict-gguf] {variant}: {len(preds)} predictions, {valid} valid JSON")


if __name__ == "__main__":
    main()

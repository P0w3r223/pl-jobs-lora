"""HF-model inference for the base and the QLoRA adapter (S5; ADR-0006).

Runs the frozen test set through the 4-bit Bielik base — with the adapter disabled (``base``
variants) and enabled (the ``-lora`` variant) — and emits ADR-0003 prediction rows for the pure
scorer, so the fine-tune drops into the same comparison table as the API baselines. Fairness: the
prompt (``build_messages``) and the parser (``parse_output``) are the shared ones, decoding is
greedy with the token cap matched to ``eval.max_tokens``, and base vs adapter differ by exactly one
toggle on an otherwise-identical load.

transformers/peft are imported lazily (Colab-only GPU stack, ADR-0004); a ``generate_fn`` seam lets
the assembly run offline in tests. Predictions echo prose and are gitignored; only the report is
committed.

    # on Colab (GPU), after training / with the adapter on HF:
    python -m pl_jobs_lora.inference.predict_hf --base --lora
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pl_jobs_lora import resume
from pl_jobs_lora.config import Config, load_config
from pl_jobs_lora.dataset.collect import DevExample
from pl_jobs_lora.eval.baselines import to_dev_examples, write_predictions
from pl_jobs_lora.eval.prompt import build_messages, parse_result
from pl_jobs_lora.normalize import load_tech_aliases
from pl_jobs_lora.train.qlora import resolve_base

_ROOT = Path(__file__).resolve().parents[3]
_PROCESSED = _ROOT / "data" / "processed"
_PRED_DIR = _ROOT / "results" / "eval" / "predictions"


def _read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_base(cfg: Config):
    """Load the 4-bit base + tokenizer (same quant config as training, ADR-0006). Needs a GPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    t, base = cfg.train, resolve_base(cfg)
    compute_dtype = torch.bfloat16 if t.compute_dtype == "bf16" else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type=t.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=t.bnb_4bit_use_double_quant, bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(base.hf_repo)
    model = AutoModelForCausalLM.from_pretrained(
        base.hf_repo, quantization_config=quant, device_map="auto"
    )
    model.eval()
    return model, tokenizer


def attach_adapter(model, adapter_repo: str):
    """Load the QLoRA adapter onto the already-loaded base (the adapter is the only variable)."""
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_repo)


def hf_generate(model, tokenizer, messages: list[dict], max_new_tokens: int) -> str:
    """Greedy generation of one completion; returns the new text only (parity with the API cap)."""
    import torch

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)


def run_inference(
    cfg: Config, eval_set: list[DevExample], shots: list[DevExample], *,
    mode: str, generate_fn, on_prediction=None,
) -> list[dict]:
    """Generate + parse predictions for one shot-mode. ``generate_fn(messages) -> raw`` is injected
    (the HF model in production, a fake in tests). No token counts — a local run is ~$0.

    ``on_prediction`` receives each row as it is produced, so a hosted-GPU session that disconnects
    mid-variant loses one record rather than the whole variant."""
    n_shots = cfg.eval.few_shot_examples if mode == "few" else 0
    alias_index = load_tech_aliases()
    preds: list[dict] = []
    for ex in eval_set:
        messages = build_messages(ex, shots, n_shots=n_shots)
        t0 = time.perf_counter()
        raw = generate_fn(messages)
        latency = time.perf_counter() - t0
        result = parse_result(raw, alias_index)
        pred = {
            "offer_id": ex.offer_id, "valid": result.valid, "parsed": result.parsed,
            "failure": result.failure, "raw": raw,
            "latency_s": round(latency, 3),
            "max_tokens": cfg.eval.max_tokens,   # the cap this row decoded under
        }
        if on_prediction is not None:
            on_prediction(pred)
        preds.append(pred)
    return preds


def _hf_generate_factory(tokenizer, max_new_tokens: int):
    """Build a ``generate_fn`` bound to a loaded model (the production seam)."""
    def factory(model):
        return lambda messages: hf_generate(model, tokenizer, messages, max_new_tokens)

    return factory


# A current prediction row carries the failure taxonomy (ADR-0003) and the cap it decoded under.
_CURRENT_ROW_KEYS = ("failure", "raw")


def _run_variant(
    cfg: Config, variant: str, eval_set: list[DevExample], shots: list[DevExample], *,
    mode: str, generate_fn, pred_dir: Path, fresh: bool,
) -> list[dict]:
    """One variant, resumably — see :mod:`pl_jobs_lora.resume`.

    This runs on a hosted GPU whose session can end without warning and whose quota is finite, so a
    disconnect at record 130 must cost one record rather than the variant. Rows that decoded under a
    different ``eval.max_tokens`` are outstanding for the same reason they are in the GGUF runner:
    keeping them would leave one file averaging two decoding budgets.
    """
    path = pred_dir / f"{variant}.jsonl"
    done = {} if fresh else resume.load_completed(
        path, required_keys=_CURRENT_ROW_KEYS,
        matches=lambda row: row.get("max_tokens") == cfg.eval.max_tokens,
    )
    todo = [ex for ex in eval_set if ex.offer_id not in done]

    existing = list(resume.load_completed(path).values())
    if todo and len(done) < len(existing):
        caps = sorted({row.get("max_tokens") for row in existing} - {None})
        suffix = ".cap" + "-".join(str(c) for c in caps) if caps else ".pre-taxonomy"
        saved = resume.backup_once(path, suffix)
        if saved is not None:
            print(f"[predict-hf] superseded rows saved to {saved.name}", flush=True)

    if todo:
        print(f"[predict-hf] {variant}: {len(done)} cached, {len(todo)} to run", flush=True)
        with resume.append_sink(path, done) as sink:
            run_inference(cfg, todo, shots, mode=mode, generate_fn=generate_fn, on_prediction=sink)

    # Deterministic file content once complete: the append order of a resumed run reflects when the
    # session dropped, not the data.
    preds = [done[ex.offer_id] for ex in eval_set if ex.offer_id in done]
    if len(preds) == len(eval_set):
        write_predictions(variant, preds, pred_dir)
    return preds


def run_predictions(
    cfg: Config, *, base: bool = True, lora: bool = False,
    processed_dir: Path = _PROCESSED, pred_dir: Path = _PRED_DIR, limit: int = 0,
    load_fn=load_base, attach_fn=attach_adapter, generate_factory=None, fresh: bool = False,
) -> dict[str, list[dict]]:
    """Emit prediction files for the requested variants over the frozen test set. Needs a GPU.

    ``load_fn``/``attach_fn``/``generate_factory`` are injectable seams so the fairness-critical
    wiring (base-before-adapter ordering, variant naming, one 4-bit load with only the adapter
    toggled) is exercised offline in tests without transformers/peft.

    **Resumable**: each row is appended as it is produced and a restart skips what is already on
    disk, so re-running after a dropped session finishes the work instead of repeating it.
    """
    base_key = resolve_base(cfg).key
    test = _read_jsonl(processed_dir / "test.jsonl")
    train = _read_jsonl(processed_dir / "train.jsonl")
    shots = to_dev_examples(train[: cfg.eval.few_shot_examples])
    eval_set = to_dev_examples(test[:limit] if limit else test)

    model, tokenizer = load_fn(cfg)
    factory = generate_factory or _hf_generate_factory(tokenizer, cfg.eval.max_tokens)
    out: dict[str, list[dict]] = {}

    if base:
        gen = factory(model)  # adapter disabled — the untuned base
        for mode in ("zero", "few"):
            variant = f"{base_key}__{mode}"
            out[variant] = _run_variant(
                cfg, variant, eval_set, shots, mode=mode, generate_fn=gen,
                pred_dir=pred_dir, fresh=fresh,
            )
    if lora:
        gen = factory(attach_fn(model, cfg.hf.adapter_repo))  # same 4-bit weights, adapter on
        variant = f"{base_key}-lora__zero"
        out[variant] = _run_variant(
            cfg, variant, eval_set, shots, mode="zero", generate_fn=gen,
            pred_dir=pred_dir, fresh=fresh,
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="HF base/LoRA inference over the test set (S5).")
    ap.add_argument("--base", action="store_true", help="predict the untuned base (zero + few)")
    ap.add_argument("--lora", action="store_true", help="predict the QLoRA adapter (zero-shot)")
    ap.add_argument("--limit", type=int, default=0, help="cap test examples for a smoke run")
    ap.add_argument(
        "--fresh", action="store_true",
        help="recompute every record instead of resuming from the predictions file",
    )
    args = ap.parse_args()
    if not (args.base or args.lora):
        ap.error("pass --base and/or --lora")

    cfg = load_config()
    out = run_predictions(
        cfg, base=args.base, lora=args.lora, limit=args.limit, fresh=args.fresh,
    )
    for variant, preds in out.items():
        valid = sum(int(p["valid"]) for p in preds)
        print(f"[predict] {variant}: {len(preds)} predictions, {valid} valid JSON")


if __name__ == "__main__":
    main()

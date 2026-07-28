"""QLoRA fine-tune of Bielik-1.5B: zero-shot completion SFT (S5; ADR-0006).

Colab-only trainer — the GPU stack (transformers/peft/bitsandbytes/trl) lives in
``requirements-train.txt`` and is imported lazily inside ``run_training`` so the pure parts here
(SFT formatting, the temporal dev split, completion masking) stay importable and testable on the
local CPU ``.venv`` (ADR-0004 dependency split).

Method (ADR-0006): each record becomes exactly the message list the zero-shot eval prompt produces
(``build_messages(ex, [], n_shots=0)``) with the gold JSON appended as the assistant turn; labels
are masked up to the generation prompt so only the gold-JSON tokens are trained — teaching the stop
token directly, which is the base model's failure (it rambles instead of stopping). Every gold
target is asserted to round-trip through ``JobPosting`` at prep time, so the model only ever learns
schema-valid output. The adapter is pushed to HF Hub; the untuned base is evaluated from the *same*
4-bit load with the adapter disabled, so the adapter-vs-base comparison isolates one variable.

    # on Colab, after `pip install -r requirements-train.txt`:
    python -m pl_jobs_lora.train.qlora --push
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pl_jobs_lora.config import Config, ModelCandidate, load_config
from pl_jobs_lora.dataset.collect import DevExample
from pl_jobs_lora.eval.prompt import build_messages
from pl_jobs_lora.schema import JobPosting

_ROOT = Path(__file__).resolve().parents[3]
_PROCESSED = _ROOT / "data" / "processed"          # frozen S2 train/test (gitignored)
_OUT = _ROOT / "results" / "train"                 # adapter + logs (gitignored)


def _read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def resolve_base(cfg: Config) -> ModelCandidate:
    """The HF base to fine-tune (ADR-0006: the probe winner, by config key)."""
    for c in cfg.models:
        if c.key == cfg.train.base:
            return c
    raise ValueError(f"train.base={cfg.train.base!r} not in models.candidates")  # defensive


# -- pure SFT formatting (no model, no tokenizer) --------------------------------------------------

def to_sft_example(record: dict) -> dict:
    """One record -> {prompt_messages, completion}. Zero-shot prompt (ADR-0006), gold-JSON target.

    The completion is the *same* serialization the collector and few-shot exemplars already use, and
    every gold is validated against ``JobPosting`` here so a malformed target fails loudly at prep
    time rather than silently teaching the model bad structure.
    """
    gold = record["gold"]
    JobPosting.model_validate(gold)  # raises on malformed gold; the target must be schema-valid
    messages = build_messages(DevExample(**record), [], n_shots=0)
    return {"prompt_messages": messages, "completion": json.dumps(gold, ensure_ascii=False)}


def temporal_dev_split(records: list[dict], dev_fraction: float) -> tuple[list[dict], list[dict]]:
    """Split off the NEWEST ``dev_fraction`` by publication date as an inner dev set (ADR-0006).

    Mirrors the ADR-0002 temporal discipline: dev is the most recent slice, so checkpoint/hyper
    selection never peeks at the future the frozen test set holds. Deterministic (stable sort);
    ``None`` dates sort oldest.
    """
    def _key(r):
        return (r.get("pub_date") is not None, r.get("pub_date") or "")

    ordered = sorted(records, key=_key)
    n_dev = min(max(1, round(len(ordered) * dev_fraction)), len(ordered) - 1)  # keep >=1 train
    return ordered[:-n_dev], ordered[-n_dev:]


def completion_labels(prompt_len: int, input_ids: list[int]) -> list[int]:
    """Mask the prompt with -100 up to ``prompt_len``; keep the ids after (train on completion)."""
    return [-100] * prompt_len + list(input_ids[prompt_len:])


def encode_example(tokenizer, sft: dict, max_seq_len: int) -> dict:
    """Tokenize one SFT example with the model's chat template; keep the completion whole.

    Uses ``tokenizer.apply_chat_template`` so Bielik's own special/EOS tokens are used (never
    hand-rolled), and asserts the prompt is a token-exact prefix of the full sequence so a chat
    template that breaks that assumption fails loudly instead of silently mis-masking every example.
    On overflow only the posting (the last user message) is shortened — never the completion or the
    fixed system/schema/assistant-header scaffold — so the JSON target is always trained whole
    (ADR-0006). ``max_seq_len`` is sized to ~p99 on Colab (Step 0), so this rarely fires.
    """
    def _ids(messages, *, with_completion):
        msgs = messages + (
            [{"role": "assistant", "content": sft["completion"]}] if with_completion else []
        )
        return tokenizer.apply_chat_template(msgs, add_generation_prompt=not with_completion)

    messages = sft["prompt_messages"]
    prompt_ids = _ids(messages, with_completion=False)
    full_ids = _ids(messages, with_completion=True)
    assert full_ids[: len(prompt_ids)] == prompt_ids, "chat template broke the prompt prefix"

    posting = messages[-1]["content"]
    while len(full_ids) > max_seq_len and posting:
        posting = posting[: -max(1, len(posting) // 10)]  # shrink the posting tail, re-template
        messages = [*messages[:-1], {**messages[-1], "content": posting}]
        prompt_ids = _ids(messages, with_completion=False)
        full_ids = _ids(messages, with_completion=True)

    return {
        "input_ids": full_ids,
        "labels": completion_labels(len(prompt_ids), full_ids),
        "attention_mask": [1] * len(full_ids),
    }


# -- Colab GPU training (lazy imports; requires requirements-train.txt) ----------------------------

def run_training(cfg: Config, *, processed_dir: Path = _PROCESSED, out_dir: Path = _OUT,
                 push: bool = False) -> Path:
    """Fine-tune the 4-bit base with a LoRA adapter and save (optionally push) it — needs a GPU.

    Selection: best checkpoint by dev loss (the 57-record dev set is small, so loss is the stable
    signal; a field-F1 epoch sweep via ``inference.predict_hf`` on dev is the notebook's job).
    Every knob comes from ``cfg.train`` — nothing hardcoded (ADR-0006).
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    from pl_jobs_lora import tracking

    t = cfg.train
    set_seed(t.seed)
    base = resolve_base(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base.hf_repo)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = _read_jsonl(processed_dir / "train.jsonl")
    train_recs, dev_recs = temporal_dev_split(records, t.dev_fraction)

    def encode(records):
        rows = [encode_example(tokenizer, to_sft_example(r), t.max_seq_len) for r in records]
        return Dataset.from_list(rows)

    train_ds, dev_ds = encode(train_recs), encode(dev_recs)

    compute_dtype = torch.bfloat16 if t.compute_dtype == "bf16" else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type=t.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=t.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base.hf_repo, quantization_config=quant, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=t.lora_r, lora_alpha=t.lora_alpha, lora_dropout=t.lora_dropout,
        target_modules=t.target_modules, bias="none", task_type="CAUSAL_LM",
    ))

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=t.epochs, learning_rate=t.learning_rate, lr_scheduler_type=t.lr_scheduler,
        warmup_ratio=t.warmup_ratio, per_device_train_batch_size=t.per_device_batch_size,
        per_device_eval_batch_size=t.per_device_batch_size,
        gradient_accumulation_steps=t.grad_accum_steps,
        fp16=(t.compute_dtype == "fp16"), bf16=(t.compute_dtype == "bf16"),
        gradient_checkpointing=True, optim="paged_adamw_8bit", seed=t.seed,
        eval_strategy="epoch", save_strategy="epoch", logging_steps=10,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        report_to=[],
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, padding="longest"),
    )

    import mlflow

    with tracking.start_run("qlora-bielik-1.5b", cfg):
        mlflow.log_params({
            "base": base.key, "lora_r": t.lora_r, "lora_alpha": t.lora_alpha,
            "lora_dropout": t.lora_dropout, "target_modules": t.target_modules,
            "epochs": t.epochs, "lr": t.learning_rate, "max_seq_len": t.max_seq_len,
            "n_train": len(train_recs), "n_dev": len(dev_recs),
        })
        result = trainer.train()
        mlflow.log_metric("train_loss", result.training_loss)
        mlflow.log_metric("eval_loss", trainer.evaluate()["eval_loss"])

    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    if push:
        trainer.model.push_to_hub(cfg.hf.adapter_repo, private=True)
        tokenizer.push_to_hub(cfg.hf.adapter_repo, private=True)
    return adapter_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="QLoRA fine-tune of Bielik-1.5B (S5; ADR-0006).")
    ap.add_argument("--push", action="store_true", help="push the adapter to HF Hub after training")
    args = ap.parse_args()
    cfg = load_config()
    adapter = run_training(cfg, push=args.push)
    print(f"[train] adapter saved -> {adapter}" + (" (pushed to HF)" if args.push else ""))


if __name__ == "__main__":
    main()

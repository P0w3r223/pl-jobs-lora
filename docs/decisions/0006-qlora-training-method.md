# ADR-0006 — QLoRA training method: zero-shot completion SFT of Bielik-1.5B

Date: 2026-07-28
Status: accepted
Author: P0w3r223
Related to: docs/decisions/0001-base-model-selection.md, 0003-evaluation-methodology.md,
            0004-training-infra.md

---

## Context

S5 turns the frozen 568/142 prose→JSON dataset into a QLoRA fine-tune of
Bielik-1.5B-v3.0-Instruct that maximizes schema-valid, field-correct extraction, scored by
the fixed pure scorer (ADR-0003) on the untouched test set. The gap to close (ADR-0001):
Bielik zero-shot emits *no* valid JSON and rambles; few-shot reaches ~0.32 field F1 but needs
exemplars and more tokens. `tech_expected` is the weakest field (~0.13). Infra (Colab GPU, HF-Hub
adapter, hosted MLflow, local/Colab dependency split) is fixed by ADR-0004; the schema, shared
prompt, and scorer are fixed for eval fairness. This ADR decides only the training *method*.

## Options

### SFT data formatting
- **A — Zero-shot, completion-only mask (chosen).** Each record becomes exactly the message list
  `build_messages(ex, [], n_shots=0)` yields, with the raw gold JSON appended as the final assistant
  turn. The full conversation is serialized via `tokenizer.apply_chat_template` (never hand-rolled),
  so Bielik's special tokens/EOS match generation. Labels are masked (`-100`) up to the generation
  prompt; only the gold-JSON tokens and the assistant terminator are trained. Target string is
  `json.dumps(gold, ensure_ascii=False)` — the same serialization the collector and few-shot
  exemplars already use — asserted to round-trip through `JobPosting` at prep time.
- **B — Few-shot training.** Rejected: +500–1000 tok/example, exemplar pattern-copying risk,
  redundant (fine-tune AND prompt), worsens truncation. No accuracy upside.

### LoRA config
- **A — r=16, alpha=32, dropout=0.1, `target_modules="all-linear"` (chosen).** NF4 4-bit +
  double-quant; compute dtype fp16 (free-Colab T4/P100 have no bf16), parametrized by GPU;
  `prepare_model_for_kbit_training` + gradient checkpointing.
- **B — r=8, alpha=16, attention-only.** Fallback if A overfits on dev; excluding the MLP
  contradicts the QLoRA all-linear finding and risks underfitting `tech_expected`.

### Training regime
- **A — inner temporal dev split, short run, dev-selected checkpoint (chosen).** Newest ~10% of
  the 568 → dev (~511/57). 3 epochs; LR 2e-4 cosine, `warmup_ratio` 0.05; `paged_adamw_8bit`;
  effective batch 16 (`per_device` 4 × `grad_accum` 4, ~96 steps); `max_seq_len` set from a
  real-tokenizer measurement to cover ~p99 with completion-preserving prose truncation;
  `packing=False`; fixed seed. Checkpoint + hyperparameters chosen by dev field-F1 (small
  epochs×eff-batch sweep); test touched once.
- **B — no dev split, fixed epochs.** Rejected: leaves checkpoint/epoch selection with nowhere to
  look but test — an eval-fairness violation.

## Decision

Adopt A/A/A: zero-shot completion-only SFT, r=16/α=32/dropout=0.1 over all-linear NF4 4-bit, with
an inner temporal dev split governing all selection so the 142-record test set is scored exactly
once. This mirrors the zero-shot eval prompt byte-for-byte (parity), directly trains the stop token
that fixes the base model's rambling, and follows the QLoRA all-linear finding while
over-regularizing (dropout 0.1, few epochs, modest rank, base frozen) against the small-data
overfit risk. All knobs go into a new `configs/config.yaml` `train:` block — no hardcoded values.

## Consequences

- **New code:** a Colab-only training entrypoint (`train/qlora.py`, imports from
  `requirements-train.txt` only) and `inference/predict_hf.py`. `predict_hf` reuses `build_messages`
  + `parse_output` for parity and emits the ADR-0003 prediction rows `{offer_id, valid, parsed,
  latency_s}` (no token counts → ~$0 local) as `results/eval/predictions/bielik-1.5b-lora__zero.jsonl`.
  It also produces `bielik-1.5b__zero` and `bielik-1.5b__few` from the *same 4-bit base with the
  adapter disabled*, so the adapter-vs-base comparison isolates exactly one variable.
- **Decoding parity (fairness):** greedy (`do_sample=False`), `max_new_tokens` matched to the eval
  cap (`eval.max_tokens`), identical normalization via `parse_output`. Any decoding drift vs the API
  baselines invalidates the comparison.
- **MLflow:** log LoRA/quant config, per-epoch train+dev loss, dev field-F1, and the final test
  report — consistent with P1.
- **Optional CPU-latency variant:** `inference/predict_gguf.py` reuses the ADR-0001 probe machinery
  to time the base model on CPU (llama.cpp), for the latency column only — the adapter is not
  GGUF-converted.
- **What we give up:** direct comparability to the *few-shot* base on an identical prompt (we
  compensate by reporting base-zero, base-few, api-zero, api-few alongside lora-zero); and the 57
  dev records are held out of training.
- **Revisit when:** dev field-F1 plateaus far below the API baseline (try Option-B smaller rank or
  more epochs), the dataset grows materially (raise rank/epochs), or the base model changes.

## Risks

- **Overfitting on 568 examples.** Mitigations already in the decision: base frozen (LoRA-only),
  r=16, dropout 0.1, 3 epochs / ~96 steps, effective batch 16, dev-selected checkpoint. Watch the
  train↔dev loss gap in MLflow; fall back to r=8 attention-only if it diverges.
- **JSON-validity vs field-accuracy.** No training-time conflict: cross-entropy on well-formed gold
  jointly drives both, and validity (the fixed JSON skeleton) is learned in the first epoch. The
  real hazard is the scorer's empty-set==empty-set → 1.0 default (`scoring.py`): a model that leaves
  sparse fields (`tech_optional`, often empty) blank earns "free" F1. This is fair (same scorer for
  all variants) but the report must not read blank-field F1 as competence — `tech_expected` F1 is
  the honest swing metric.
- **Sparse-salary label noise.** Salary is 31% covered and, by the leakage guard, usually *not in
  the prose the model sees* — so those targets ask the model to emit salary it cannot observe.
  Kept as-is for gold consistency; salary is small in the CE loss and scored separately
  (currency/kind/amount), so it does not pollute tech/seniority/work-mode F1. Expectation set
  honestly: QLoRA will not "fix" salary — it is a data-availability ceiling, not a training failure.
- **Eval fairness.** Guaranteed by: identical prompt (`build_messages`) + scorer + frozen test set;
  test touched once (all tuning on the inner dev split); base and LoRA loaded identically at 4-bit
  with only the adapter toggled; matched greedy decoding and token cap. The probe's Q8_0/CPU base
  numbers stay as context, but the headline adapter attribution uses the identical 4-bit GPU load.

## Colab Step 0 (before writing final `max_seq_len`)

Tokenize the 568 train records with the real Bielik tokenizer, set `train.max_seq_len` to cover
~p99 of prompt+completion, and verify the completion (gold JSON) is never truncated. The config
default (2048) is an estimate from character lengths, to be confirmed from data.

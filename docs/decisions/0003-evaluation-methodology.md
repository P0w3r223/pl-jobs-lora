# ADR-0003 — Per-field metrics with a pure, model-free scorer

Date: 2026-07-27
Status: accepted
Author: P0w3r223

---

## Context

Every variant — zero-shot API, few-shot API, base model untuned, the QLoRA model, and optionally a
GGUF CPU build — must be scored by the *same* harness on the *same* frozen test set and compared on
accuracy × cost × latency. The fields are incommensurable (a set-valued tech list vs a numeric
salary), so one averaged number would hide where the LoRA actually wins or loses.

## Options

- **A — One averaged accuracy number.** Rejected: averages across incommensurable fields; opaque.
- **B — Per-field metrics + decoupled scoring (chosen).**

## Decision

Option B.

- **Per-field metrics** (both sides normalized via the vendored it-job-radar functions, ADR-0002):
  - `tech_expected`/`tech_optional`: set precision/recall/**F1** after `normalize_technology`
    (order-invariant, alias-fair).
  - `seniority`, `work_mode`: multi-label set → exact-set match + micro-F1.
  - `salary`: split scoring — `currency` exact, `kind` (b2b/employment) exact, bounds within a
    tolerance band; reported separately because prose-recoverability is weakest here.
  - `JSON validity`: first-class metric (parses + satisfies the schema) — exposes a base model's
    malformed output.
- **Decoupling:** each variant emits `predictions/{variant}.jsonl`; the scorer is a **pure function**
  `(predictions, gold) → scores` that never loads a model. Inference runs wherever the model lives
  (API locally, base/LoRA on Colab, GGUF locally); scoring is identical and offline.
- **Fairness:** identical prompt template + `JobPosting.prompt_schema()` for the API baselines and
  the base model — the only variable is the model.
- **Cost/latency:** API = tokens × current price (pull model IDs/pricing at implementation, not from
  memory) + measured p50/p95 latency; LoRA/GGUF = ~0 marginal + measured latency. Every run versioned
  to `results/`.

## Consequences

- The headline question the report answers: *does a 1.5B local LoRA match a frontier API on this
  narrow task at a fraction of the cost/latency?*
- A deliberately broken prompt must show visibly lower scores (regression is visible), the same
  property P2/P3's harnesses guarantee.

## Implementation (S4)

Resolved 2026-07-28. `eval/scoring.py` is the pure per-field scorer (built in S1 for the probe);
S4 adds `eval/baselines.py` (zero-/few-shot API baselines over the frozen test set), `eval/pricing.py`
(pure cost/latency economics), and `eval/report.py` (pure aggregation → `results/eval/report.{json,md}`),
driven by `eval/run.py` (`--baselines` paid, `--report` offline). The **baseline API model is
`claude-haiku-4-5`** (cheap frontier — sharpens the "cheap API vs own LoRA" cost axis); model IDs and
per-MTok pricing live in `configs/config.yaml`, pulled at implementation time, not from memory. The
baseline generates **plain text** with the shared probe prompt (no forced tool call) so JSON validity
stays a metric it can fail; the Anthropic client is a lazy `api` extra, so tests/CI stay offline. The
report merges any `predictions/{variant}.jsonl`, so the base/QLoRA runs from S5 drop into the same
table. The paid baseline run and the QLoRA adapter (S5) are pending.

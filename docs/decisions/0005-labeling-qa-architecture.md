# ADR-0005 — Labeling-QA: adjudication authority and the proposer instrument

Date: 2026-07-28
Status: accepted
Author: Piotr Cząstkiewicz
Related to: docs/decisions/0002-dataset-and-labeling.md

---

## Context

ADR-0002 chose triangulated labels (platform-gold + LLM-from-prose + human adjudication) but left two
things open for S3: **who authors the human-checked gold sample**, and **which model proposes labels
from prose at full scale**. S3 runs this QA over the frozen S2 set (710 records, platform-gold labels)
and emits an agreement report. Two facts shape the choice: the chosen base model is small
(Bielik-1.5B, few-shot field-F1 ~0.32 on the S1 probe — a weak QA instrument), and the human is a
single annotator on a portfolio (no inter-annotator pool). The hard constraint is that raw postings
never leave the machine; prose is PII-free (the `applying` block is dropped) and the technologies
label-widget is already excluded from prose by the S2 leakage guard.

## Options

- **Full-scale proposer — A: local Bielik-1.5B few-shot (chosen).** Offline, zero egress, cacheable
  like the probe slice. A low LLM↔platform number is honestly a *small-model prose-recoverability
  floor* that motivates QLoRA, not a claim about label quality.
- **Full-scale proposer — B: a strong API / larger-local model.** Makes disagreement more diagnostic
  but costs egress (API) or many CPU-hours + a large download (local), and conflates the QA proposer
  with the S4 subject model. Rejected for the full scale because the *human* leg — not the proposer —
  is what catches platform gaps.
- **Adjudication — 1: human-from-scratch on the sample.** Strongest rebuttal to "you just trusted an
  LLM," but the most human time.
- **Adjudication — 2: strong-LLM first pass + human spot-check.** Cheapest, but weakens the rebuttal —
  the "human gold" is really strong-LLM gold.
- **Adjudication — 3: hybrid with an API arbiter (chosen).** A strong API model (distinct from the
  full-scale proposer) does a first pass on the sample; the human adjudicates **every contested cell**
  (any field where proposer/platform/arbiter are not unanimous) and spot-checks the unanimous rest.

## Decision

**Full-scale proposer = local Bielik-1.5B few-shot** (Option A), reported as a prose-recoverability
floor. **Adjudication = hybrid with an API arbiter** (Option 3): the arbiter is `claude-opus-4-8`,
distinct from the Bielik proposer so LLM↔human is not self-agreement. The sample is **72 records drawn
from all 710**, stratified by LLM↔platform disagreement and seeded. The `proposer_backend: gguf|api`
config seam is kept, but **only `gguf` is implemented now** (YAGNI); selecting `api` fails fast at
config load.

## Consequences

- **Egress:** the arbiter is the single egress point — **≤80 PII-free prose snippets** sent to the
  Anthropic API, hard-capped in config (`arbiter_max_snippets`). Approved by the maintainer.
  Determinism comes from the task shape and structured outputs, not a `temperature` param (Opus 4.8
  rejects sampling params).
- **Report legs:** LLM↔platform over all 710 (minus the few-shot shots, exact `n` in metadata);
  LLM↔human and platform↔human over the 72. Triangulation buckets each field into *all-agree /
  platform-gap-caught (LLM==human≠platform) / LLM-error (platform==human≠LLM) / all-differ*. Beyond
  reused per-field F1, the report adds raw agreement rate and **Cohen's κ** for the categorical fields
  (seniority, work_mode), modeling each side's normalized field value as one category.
- **Purity:** the agreement math (`dataset/agreement.py`) and the sampler are pure and offline;
  model + network + files live only in `dataset/labeling_qa.py`. Both label sides normalize through
  the vendored functions before comparison (ADR-0003), so the comparison stays fair.
- **`validated_f1` is a soft flag** surfaced in the report — never a pass/fail gate.
- **Optional deps:** the Anthropic client ships as an optional `api` extra and is imported lazily, so
  the core `.venv`, pytest, and CI stay fully offline. CI runs the agreement layer on
  `data/fixtures/` only.

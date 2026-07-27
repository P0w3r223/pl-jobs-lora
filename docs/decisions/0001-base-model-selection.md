# ADR-0001 — Base-model selection by empirical probe, not reputation

Date: 2026-07-27
Status: accepted
Author: P0w3r223 + Claude

---

## Context

The QLoRA fine-tune needs a small (~1.5B) open base model. Two candidates: **Bielik-1.5B**
(Polish-specialized) and **Qwen2.5-1.5B-Instruct** (strong multilingual, Apache-2.0). Both fit a
free T4/P100 at 4-bit. The plan forbids pre-picking one — the choice must be defensible on an
interview ("why this base?").

## Options

- **A — Empirical probe on a held-out dev slice (chosen).** Run *both* base models zero-shot +
  few-shot over ~25 dev offers through the P4 harness; score per-field accuracy, JSON-validity
  rate, output token economy / latency, and confirm license. Pick the winner, record the numbers.
- **B — Pick by reputation.** Zero cost, but indefensible and violates the "decide in an ADR"
  constraint.

## Decision

Option A. The probe reuses the harness we build anyway (ADR-0003) and its result *is* a portfolio
artifact. It measures the base model's steerability toward our JSON schema as a proxy for how well
QLoRA will land — it selects the better **starting point**, not a final quality claim.

## Consequences

- S1 ends with the base model chosen from data, numbers recorded here.
- The probe runs on the **local CPU via GGUF q4** (user decision) for a GPU-free, reproducible S1;
  the exact GGUF repo/file per candidate is resolved during the probe and written to
  `configs/config.yaml` (`models.candidates[].gguf_repo/gguf_file`). If a candidate lacks a usable
  GGUF build (Bielik may need conversion), that one candidate's probe falls back to Colab.
- If the probe is inconclusive on license/benchmarks, spawn a researcher for external evidence
  before committing.

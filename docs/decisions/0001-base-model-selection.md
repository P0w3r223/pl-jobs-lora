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

## Result (2026-07-27)

Probe run on local CPU, 25-offer live dev slice (2 few-shot exemplars, **n=23** eval),
matched **Q8_0** for both candidates, pure ADR-0003 scorer. Winner by
(mean field F1 + JSON-validity), tie-break lower p50 latency:

| variant | field F1 | JSON valid | salary cur/kind/amt | p50 s | out tok |
|---|---|---|---|---|---|
| qwen2.5-1.5b / zero | 0.224 | 0.96 | 0.26 / 0.17 / 0.22 | 31.7 | 417 |
| qwen2.5-1.5b / few | 0.322 | 0.87 | 0.74 / 0.91 / 0.70 | 15.8 | 233 |
| bielik-1.5b / zero | 0.000 | 0.00 | 0.74 / 0.96 / 0.74 | 65.2 | 770 |
| **bielik-1.5b / few** | 0.316 | **0.91** | 0.74 / 0.96 / 0.74 | 23.5 | 203 |

**Chosen base: Bielik-1.5B-v3.0-Instruct**, run few-shot. Rationale:

- Combined criterion: bielik/few 1.229 > qwen/few 1.192 — Bielik trades ~0.006 field F1 for
  +0.04 JSON-validity and higher salary/kind accuracy.
- **Zero-shot Bielik never emits valid JSON** (0.00) and rambles (770 tok, 65 s) — a real finding:
  it needs few-shot steering, which QLoRA supersedes. This is the *starting point* being measured,
  not a quality verdict.
- Both bases are weak on `tech_expected` (F1 ~0.13) and `title` — the gap QLoRA must close; sets S4
  expectations.

## Consequences

- S1 ends with the base model chosen from data; numbers recorded above.
- **Quantization deviation from plan:** the plan said GGUF **q4**; Bielik ships **no q4** build
  (only fp16 / Q8_0). For a fair comparison both candidates were probed at **matched Q8_0**. Q8_0 is
  lossless-adjacent, so the ranking is if anything *more* trustworthy than q4 would be. Config
  comment + README note this.
- Probe ran on **local CPU via GGUF** (llama-cpp-python) for a GPU-free, reproducible S1; the exact
  GGUF repo/file per candidate lives in `configs/config.yaml`
  (`models.candidates[].gguf_repo/gguf_file`). On Windows the prebuilt CPU wheel is required (no
  compiler) — README documents the index.
- License confirmed usable for both (Apache-2.0 Qwen; Bielik SpeakLeash/Apache-2.0). Probe was
  conclusive — no external-evidence researcher needed.

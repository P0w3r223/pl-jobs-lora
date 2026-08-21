# ADR-0001 — Base-model selection by empirical probe, not reputation

Date: 2026-07-27
Status: accepted
Author: P0w3r223

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

Both the salary columns and the `field F1` definition were **corrected on 2026-08-21** — see the
two amendments below. The table as published on 2026-07-27 credited a model for saying nothing,
and averaged in a label the model's input does not contain.

| variant | field F1 | JSON valid | salary detect | salary cur/kind/amt | p50 s | out tok |
|---|---|---|---|---|---|---|
| qwen2.5-1.5b / zero | 0.299 | 0.96 | 0.26 | 0.67 / 0.00 / 0.00 | 31.7 | 417 |
| qwen2.5-1.5b / few | 0.429 | 0.87 | 0.70 | 0.17 / 0.00 / 0.00 | 15.8 | 233 |
| bielik-1.5b / zero | 0.000 | 0.00 | 0.00 | 0.00 / 0.00 / 0.00 | 65.2 | 770 |
| **bielik-1.5b / few** | 0.421 | **0.91** | 0.70 | 0.00 / 0.00 / 0.00 | 23.5 | 203 |

**Chosen base: Bielik-1.5B-v3.0-Instruct**, run few-shot. Rationale:

- Combined criterion: bielik/few 1.334 > qwen/few 1.299 — Bielik trades ~0.009 field F1 for
  +0.04 JSON-validity. (The original wording added "and higher salary/kind accuracy"; under the
  corrected metric that clause is false and has been struck — see the first amendment.)
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

## Amendment (2026-08-21) — the salary columns were measuring label sparsity

The scoring defect described in [ADR-0003's amendment](0003-evaluation-methodology.md) applies to
this table too, and the probe slice shows it in its purest form: **`bielik-1.5b / zero` was
published at `0.74 / 0.96 / 0.74` on salary while emitting valid JSON on 0 % of the slice.** The old
scorer credited `None == None`, only 6 of the 23 eval records carry a salary, and a model producing
nothing at all inherited the other 17.

The numbers above were regenerated from the **cached slice and the stored predictions** — no model
was re-run and no prediction changed:

```bash
.venv/Scripts/python -m pl_jobs_lora.probe --rescore   # offline; loads no model
```

**The decision stands.** `mean_field_f1` and JSON-validity are untouched by the fix, so the
combined criterion is unchanged (bielik/few 1.229 > qwen/few 1.192) and `pick_winner` still
returns `bielik-1.5b/few`. What changes is one clause of the rationale: Bielik does **not** have
higher salary accuracy. On the 6 supported records qwen/few recovers the currency once (0.17) and
Bielik never does, and *no* variant gets a single `kind` or `amount` right. The honest reading is
that **salary is unmeasurable at this slice size** — 6 records is too thin to separate two models —
and it should never have appeared as a discriminator in the rationale.

`--rescore` exists because of this: the published probe numbers must be reproducible from the
cached slice whenever the scorer changes, without re-paying for inference. It is the probe's
counterpart to `eval.run --report`.

## Amendment (2026-08-21) — `field F1` no longer averages in an unanswerable label

The headline `field F1` averaged four fields, one of which — `tech_optional` — is largely absent
from the prose the model is given. Measured model-free on the full 710-posting set, only **14–17 %**
of its gold terms occur in their own posting's text, and **75 %** of the postings that carry gold
optional terms contain none of them: the platform files those technologies in the widget that
ADR-0002's leakage guard strips out. Averaging that into a headline measures the dataset, not the
model, so `HEADLINE_FIELDS` now covers `seniority`, `work_mode` and `tech_expected`.
`tech_optional` is still scored and reported, next to its own ceiling. See ADR-0003.

The probe table above was regenerated with `probe --rescore`; every field F1 rises, because the
term being dropped was near zero for every variant.

**The decision stands, for the second time.** bielik/few 1.334 > qwen/few 1.299 — the same
ordering and very nearly the same margin (0.035 vs 0.037 before). `pick_winner` returns
`bielik-1.5b/few` under both definitions. Two independent metric corrections have now moved every
number in this table without moving its conclusion, which is a better argument for the choice than
the original numbers were.

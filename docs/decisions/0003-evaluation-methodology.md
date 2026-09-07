# ADR-0003 — Per-field metrics with a pure, model-free scorer

Date: 2026-07-27
Status: accepted
Author: Piotr Cząstkiewicz

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

## Amendment (2026-08-18) — silence must not earn credit

Found during a review of this harness as the reference for a successor extraction project.

**The defect.** The original `_score_salary` returned `{"currency": 1, "kind": 1, "amount": 1}`
whenever both sides were `None`, and `SetMetric.add` credited `exact += int(ps == gs)` for
empty-vs-empty. **107 of the 142 gold records carry no salary**, so a variant emitting nothing at
all inherited that base rate. The published table reported `bielik-1.5b-gguf__zero` at
**0.75/0.87/0.74 on salary while producing valid JSON 4.9 % of the time** — statistically
indistinguishable from the frontier baseline. That column measured the prevalence of missing
salaries, not accuracy. The module docstring's claim that an invalid prediction "scores 0 on every
field" was false: the one test guarding it used gold that *had* a salary, so it never exercised the
dominant case.

**The correction.** Four changes to `eval/scoring.py`, none of which alter what is being asked of a
model — only how the answer is counted:

1. **Detection is split from value accuracy.** `salary.detection` scores the present/absent decision
   over all records (deciding "absent" correctly is a real answer); `currency`/`kind`/`amount` are
   denominated by `support` — the 35 records whose gold has a salary. Predicting nothing now earns
   nothing on the value half.
2. **Every field reports `support`.** `exact_match` is computed over supported records only, and a
   field with no support reports `None`, rendered `-` — unmeasurable on this test set, not perfect.
3. **An invalid prediction is scored as no answer given**: no true positives, no exact match, and a
   failed detection. The docstring is now true, and a regression test exercises exactly the case the
   old one missed.
4. **Coverage is first-class.** `ScoreReport` carries `n`, `n_gold` and `coverage`; the rendered
   table shows it per variant and flags anything below 1.00. Previously a run that died at record 5
   of 142 scored its own subset while the header printed the gold count.

**Kept deliberately separate.** `dataset/agreement.py` compares two *annotations* of the same
posting, where "both legs say no salary" genuinely is agreement. That semantics now lives in a named
`salary_equal()` rather than being borrowed from a scoring internal whose asymmetry it does not want.

**Consequences.** The corrected salary figures are far worse and far more informative: the frontier
baseline recovers currency on 23 % of the salaries actually present, arrangement on 11 %, and the
amount bounds on **6 %**. Salary is the weakest part of this task by a wide margin and the old metric
hid it completely. `results/eval/report.{json,md}` and the README table were regenerated from the
unchanged prediction files; no model was re-run and no prediction changed.

**Still open, not fixed here.** No failure taxonomy or raw output is recorded per prediction, so the
cause of the 0.05 zero-shot validity cannot be recovered without re-paying for the run;
`tech_optional` (support 53/142) scores 0.01–0.02 for every variant yet carries equal weight in the
headline `field F1`; and the local side is priced `-` rather than as a number.

## Amendment (2026-08-21) — a metric that cannot be interrogated, and one with no error bars

Two of the gaps left open above are now closed. Neither changes what is asked of a model or how any
existing number is computed; both change what the harness can *tell you* about a number.

### Failure taxonomy

`valid` was a bare boolean, so the harness could report that a variant failed on 95 % of postings
without being able to say whether it produced no JSON, unclosed JSON, or JSON the schema rejected.
Those have different causes and different fixes, and recovering the difference meant paying for the
run again. The parser now returns a `ParseResult` carrying a **failure class** —
`empty_output` / `no_json_object` / `json_decode_error` / `schema_invalid` — and every writer
persists it next to the raw output. The classes live in `eval/scoring.py` rather than `eval/prompt.py`
so the offline report can tabulate them without importing the prompt's HTTP-carrying dependency chain.

Two deliberate choices. There is **no `not_an_object` class**: extraction anchors on the first `{`
and a decode anchored there yields a dict or raises, so such a class could never fire — and a
taxonomy entry that always reads zero is indistinguishable from one that is never detected. And
rows written before this existed are counted as **`unrecorded`**, never folded into the others, so
an old predictions file cannot be mistaken for one that parsed cleanly.

The report cross-tabs `json_decode_error` against the decoding cap. That separates *truncated* from
*malformed*, which is the confound the GGUF runs' 1024-token cap left unresolvable.

### Uncertainty

Every number in this report is computed over **142 records**, and the report published them as bare
points — leaving a reader to guess whether `0.39` vs `0.23` is a finding or a draw. `--report` now
resamples the test set with replacement (seeded; `scoring.bootstrap_*` in config) and reports a
percentile interval per variant.

The load-bearing part is that differences are **paired**: within one resample every variant is
scored on the *same* drawn records, so the "was this a hard draw of postings" component the
variants share cancels instead of inflating both intervals. Comparing two independent CIs by eye is
weaker — overlapping intervals routinely hide a consistently one-signed paired difference. Pairing
also falls out of a single resampling loop, so it costs nothing over the naive version.

Resampling is by **record**, never by prediction row: a posting is the unit of independence.
The reported `separated` flag means the interval on the difference excludes zero.

**Consequence on the current table.** Every gap against the best variant survives resampling except
one — the two Haiku variants tie exactly on JSON validity (1.00 each), reported as `separated: no`.
Few-shot's `+0.06` field F1 over zero-shot is separated, so it is a real effect rather than noise.

Two properties are pinned by tests because they are the ways this section could quietly lie. A
**partial** variant is scored on its own subset of every draw, so its difference from a complete
variant is between two populations, not one; such pairs are reported `n/a`, never as a tie —
otherwise a run that answered half the set would read as indistinguishable from a full one. And
the bootstrap scores duplicate prediction rows exactly as the main table's scorer does, so the
point estimate here and the point estimate there cannot disagree on the same file.

Resampling indices are derived from `random()` rather than `randrange()`: only the former's
stream is a documented cross-version guarantee, and these intervals are a committed artifact.

## Amendment (2026-08-21) — an F1 that cannot be read without its ceiling

`tech_optional` scored 0.01–0.02 for *every* variant, including the frontier baseline. A metric on
which the best available model does no better than the worst is usually not measuring the model,
and three independent checks agree that it was not:

1. The labeling-QA proposer, reading **prose only** across 708 postings, emitted a non-empty
   `tech_optional` on **27** records — against 575 for `tech_expected`. It was not guessing wrong;
   it had nothing to guess from.
2. `claude-haiku-4-5` *does* attempt the field (36 records against a support of 53) and reaches a
   precision of **0.03**, while scoring 0.29 on `tech_expected` in the same call.
3. Model-free, on the text itself: only **14–17 %** of gold `tech_optional` terms occur anywhere in
   their own posting's prose, and **75 %** of the postings carrying gold optional terms contain not
   one of them. For `tech_expected` the same measurement gives 32–35 % and 36 %.

The cause is a known design decision whose magnitude was never quantified: gold tech labels come
from theprotocol's technologies widget, and ADR-0002's leakage guard strips that widget from the
prose so the task is reading rather than copying. The labels the guard makes unanswerable stayed in
the metric anyway.

**Two changes.** `HEADLINE_FIELDS` drops `tech_optional` from the averaged `field F1` — the field is
still scored and reported, but is not treated as evidence about a model. And `eval/ceiling.py` adds
a **model-free data ceiling**: per open-vocabulary field, the share of gold terms present in the
prose at all, rendered beside the scores. It is a bound, not a target — presence is necessary for
extraction, not sufficient — and the alias-map fallback makes it conservative, so it reads as a
floor on the ceiling. Closed-vocabulary fields (`seniority`, `work_mode`) get none: their values are
expressed in free Polish, so a canonical-token search would measure the vocabulary, not the data.

**What it changes about the results.** On the test set the ceiling for `tech_expected` is **0.28**
and the best recall achieved is **0.27** — the frontier baseline is at **94 %** of what the input
makes recoverable. Read without the ceiling, 0.28 F1 looked like a weak model; read with it, the
remaining headroom on this field is mostly not there to be taken. That reframes what the QLoRA
fine-tune can be expected to win: `JSON validity` (0.05 zero-shot), `seniority` and `work_mode` are
genuinely open; `tech_*` is close to a data ceiling no fine-tune can lift.

Headline `field F1` rises for every variant because the dropped field was near zero everywhere —
`claude-haiku-4-5__few` from 0.39 to 0.51. ADR-0001's probe table was regenerated for the same
reason; the base-model decision is unchanged (see its second 2026-08-21 amendment).

**Still open.** The bootstrap covers `field F1` and `JSON valid` only, not the per-field or salary
columns; the local side is still priced `-`; and the ceiling is computed for the tech fields only,
so `seniority`/`work_mode`/`salary` have no comparable answerability bound. The taxonomy can only diagnose runs made *after* it existed — the four prediction
files on disk report `unrecorded` for all 164 of their invalid rows. And one `decode_max_tokens` is
applied to every variant when cross-tabbing truncation, which is correct only while
`probe.max_tokens` and `eval.max_tokens` agree (both 1024); if they ever diverge, the GGUF rows'
`at token cap` column would be attributed against the wrong cap.

### Amendment — 2026-08-21 (third): the truncation confound, resolved by measurement

The cross-tab above raised a question it could not answer: 13 of the 14 `json_decode_error` rows on
the few-shot GGUF variant sat at *exactly* the 1024-token cap, so "the model produced broken JSON"
and "we stopped the model mid-JSON" were the same number. The obvious reading — that the cap was
starving the run — was reinforced by the zero-shot variant, whose longest *finished* answer reached
985 tokens, 96 % of the cap.

Both GGUF variants were therefore re-run at **2048** (config carries the reasoning; 2.1× the longest
completion the model has ever finished, 2.6× the longest the API baseline produced). Doubling the
budget changed **nothing** on the few-shot variant:

| | cap 1024 | cap 2048 |
|---|---|---|
| valid JSON | 113 / 142 | **113 / 142** |
| rows at cap | 13 | **13 — the same records** |
| `invalid → valid` | — | **0** |
| `valid → invalid` | — | **0** |

So these failures are not budget-limited. The model does not terminate, and would not terminate at
any cap this project can afford to decode. That is a *negative* result worth the CPU: it closes the
confound instead of leaving every future reader of the `at token cap` column to wonder. (The
zero-shot variant is being re-measured separately — it is the case where the cap plausibly did
bind, at 36 % of rows against few-shot's 9 %.)

**Greedy decoding is deterministic, and now measured to be.** All **129** few-shot rows that ended
on their own under the old cap produced byte-identical output under the new one. This was the
argument for not re-paying for the API baselines when the cap moved — 0 of their 284 rows ever
reached 1024, so the cap never bound them — and it is no longer an argument but an observation on
129 records. It is also what makes the resume predicate safe to reason about: a row is invalidated
by a cap change only because its *own* cap is part of what it measured, not because the model might
have drifted.

**Latency carries ~10 % run-to-run drift.** The same 129 byte-identical rows were 1.08–1.14× slower
in the second run, and the inflation is flat across the whole run rather than concentrated where
other work was competing for the CPU — so it is ambient, not contamination. The report publishes
p50/p95 latency as bare points while giving `field F1` a bootstrap interval; a latency gap smaller
than about 10 % between two variants should be read as a draw, on the same "is this a finding"
discipline the uncertainty section applies to accuracy.

**Two items retired from "Still open" above.** The taxonomy is no longer confined to runs made after
it existed — `resume.load_completed(required_keys=...)` re-runs stale rows, which is how both GGUF
variants were backfilled. And `decode_max_tokens` is no longer applied globally: every writer stamps
the cap its row decoded under and the cross-tab prefers it, so the probe and eval caps may diverge
without misattributing truncation. The API rows predate that field and fall back to the config
value, which still reports them — correctly — as never truncated.

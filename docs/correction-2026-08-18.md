# Correction: salary scoring before 2026-08-18

The earlier scorer credited `None == None`, and **107 of 142 gold records carry no salary**. A model
emitting nothing at all therefore inherited that base rate: the previous table reported
`bielik-1.5b-gguf__zero` at ~~**0.75/0.87/0.74 on salary while producing valid JSON 4.9 % of the
time**~~. That number measured the prevalence of missing salaries, not accuracy.

The metric is now split, and both halves are denominated honestly:

- **`salary detect`** — the present/absent decision, over all 142 records. Correctly answering
  "no salary" is a real answer and is credited here.
- **`salary cur/kind/amt`** — accuracy over the **35** records whose gold *has* a salary. Predicting
  nothing earns nothing.

The corrected picture is much worse and much more informative: even the frontier baseline recovers
the currency on 0.23 of the salaries present, the arrangement on 0.11, and the amount bounds on
**6 %**. Salary extraction is the weakest part of this task by a wide margin, which the old metric
hid entirely. The same fix applies to set-valued fields — empty-vs-empty no longer counts as an
exact match, and a field with no support renders `-` rather than a perfect score. Regression tests
covering both cases are in `tests/test_scoring.py`.

The same defect reached the **base-model probe** (ADR-0001), where it is starker still: that table
published `bielik-1.5b / zero` at `0.74/0.96/0.74` on salary *while it emitted valid JSON on 0 % of
the slice*. `results/probe/report.{json,md}` and the ADR table were regenerated from the cached
slice via the new `probe --rescore` — the winner and the combined criterion are unchanged, one
clause of the rationale was wrong and is struck. Details in
[ADR-0001](decisions/0001-base-model-selection.md#amendment-2026-08-21--the-salary-columns-were-measuring-label-sparsity).

Zero-shot base confirms the ADR-0001 probe finding at full scale (0.05 valid vs 0.80 few-shot).
Per-field accuracy stays weak (0.30 field F1 few-shot), which is the gap QLoRA (S5) targets.


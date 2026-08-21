# Evaluation report (gold n=142)

Every variant scored by the same pure scorer on the same frozen test set (ADR-0003). API cost priced at claude-haiku-4-5 $1.0/$5.0 per MTok.

`coverage` is the fraction of the gold set the variant actually predicted — a run that died part-way scores only its own subset, and must not read as a full one. Salary is split: `detect` is the present/absent decision over all records, `cur/kind/amt` are accuracies over the records whose gold *has* a salary (support n=35), so predicting nothing earns nothing. `-` means unmeasurable on this test set, never a perfect score.

| variant | coverage | JSON valid | seniority F1 | tech F1 | work-mode F1 | salary detect | salary cur/kind/amt | field F1 | $/1k | p50 s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bielik-1.5b-gguf__few | 1.00 | 0.80 | 0.25 | 0.12 | 0.54 | 0.61 | 0.00/0.00/0.00 | 0.30 | - | 21.43 | 103.02 |
| bielik-1.5b-gguf__zero | 1.00 | 0.05 | 0.07 | 0.01 | 0.05 | 0.03 | 0.03/0.00/0.00 | 0.04 | - | 67.90 | 102.58 |
| claude-haiku-4-5__few | 1.00 | 1.00 | 0.62 | 0.28 | 0.63 | 0.78 | 0.23/0.11/0.06 | 0.51 | 4.47 | 2.25 | 4.13 |
| claude-haiku-4-5__zero | 1.00 | 1.00 | 0.44 | 0.26 | 0.60 | 0.77 | 0.23/0.11/0.06 | 0.43 | 3.34 | 2.47 | 4.68 |

## Failure taxonomy

`JSON valid` above says *how often* a variant failed; this says *how*. `at token cap` counts the `json_decode_error` rows whose output reached the decoding cap — plausibly truncated rather than malformed, the one confound the validity rate cannot separate on its own. `unrecorded` marks rows written before the parser classified its failures.

| variant | invalid | unrecorded | at token cap |
|---|---|---|---|
| bielik-1.5b-gguf__few | 29 | 29 | - |
| bielik-1.5b-gguf__zero | 135 | 135 | - |
| claude-haiku-4-5__few | 0 | 0 | - |
| claude-haiku-4-5__zero | 0 | 0 | - |

## Data ceiling (model-free)

Gold tech labels come from the platform's technologies widget, which the ADR-0002 leakage guard strips out of the prose — otherwise the task would be copying a list. This measures the consequence: the share of gold terms that appear in the posting text **at all**. It bounds recall from above, so a raw F1 on these fields cannot be read without it. Necessary, not sufficient — and matching is conservative, so treat it as a floor on the ceiling.

| field | gold terms | in prose | ceiling on recall | postings with gold | ...where no term is in the prose |
|---|---|---|---|---|---|
| tech_expected | 542 | 153 | 0.28 | 115 | 41 (0.36) |
| tech_optional | 154 | 20 | 0.13 | 53 | 40 (0.75) |

Best recall achieved, against that ceiling:

- `tech_expected`: 0.27 of a ceiling of 0.28 — **94% of what the input makes recoverable**.
- `tech_optional`: 0.02 of a ceiling of 0.13 — **15% of what the input makes recoverable**.

## Uncertainty (bootstrap)

2000 resamples of the 142 test records, drawn with replacement (seed 20260728); intervals are 95 % percentile bands. The point estimates are the numbers in the table above — the interval says how far a different sample of postings could have moved them.

| variant | field F1 | 95 % CI | JSON valid | 95 % CI |
|---|---|---|---|---|
| bielik-1.5b-gguf__few | 0.30 | [0.27, 0.34] | 0.80 | [0.73, 0.86] |
| bielik-1.5b-gguf__zero | 0.04 | [0.01, 0.07] | 0.05 | [0.01, 0.09] |
| claude-haiku-4-5__few | 0.51 | [0.47, 0.54] | 1.00 | [1.00, 1.00] |
| claude-haiku-4-5__zero | 0.43 | [0.39, 0.47] | 1.00 | [1.00, 1.00] |

### Paired differences vs `claude-haiku-4-5__few`

Where both variants answered the whole gold set, each resample scores them on the **same** drawn records, so the shared difficulty of the draw cancels. `separated` means the interval excludes zero — the gap survives resampling. Two overlapping per-variant CIs can still yield a separated paired difference; that is the point of pairing. `sign agreement` is the share of resamples landing on the observed side of zero — or, for an exactly tied difference, the share that also tie.

| variant | metric | difference | 95 % CI | sign agreement | separated |
|---|---|---|---|---|---|
| bielik-1.5b-gguf__few | mean_field_f1 | -0.20 | [-0.25, -0.16] | 1.00 | yes |
| bielik-1.5b-gguf__few | json_validity | -0.20 | [-0.27, -0.14] | 1.00 | yes |
| bielik-1.5b-gguf__zero | mean_field_f1 | -0.46 | [-0.51, -0.42] | 1.00 | yes |
| bielik-1.5b-gguf__zero | json_validity | -0.95 | [-0.99, -0.91] | 1.00 | yes |
| claude-haiku-4-5__zero | mean_field_f1 | -0.07 | [-0.10, -0.05] | 1.00 | yes |
| claude-haiku-4-5__zero | json_validity | 0.00 | [0.00, 0.00] | 1.00 | no |

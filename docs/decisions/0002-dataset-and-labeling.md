# ADR-0002 — Dataset construction and triangulated labeling QA

Date: 2026-07-27
Status: accepted
Author: P0w3r223 + Claude
Related to: docs/research/f6-data-availability.md

---

## Context

P4 needs 400–800 examples of *posting prose → JSON*, a time-based split, raw postings kept out of
the repo, and a labeling-QA loop. Constraints from the data source (verified in F6): it-job-radar's
DB stores only structured attributes, not prose; offers expire and theprotocol strips pages for
datacenter IPs, so stored URLs are not reliably re-fetchable and CI cannot fetch.

## Options

- **A — LLM-only silver labels + human check of 60–80 (the plan's literal shape).** Simplest, but
  ignores theprotocol's structured attributes — the biggest asset — leaving no independent gold and
  QA measured only on the small sample.
- **B — Triangulated labels (chosen): platform-gold + LLM-from-prose + human adjudication.**
  Reference labels come from the structured `__NEXT_DATA__` attributes (employer-entered,
  deterministic). The LLM proposes labels from **prose only** (the real task). The builder computes
  **LLM↔platform agreement at full scale** *and* **LLM↔human agreement on 60–80**, where humans also
  adjudicate LLM↔platform disagreements (catching platform gaps).

## Decision

Option B. An independent reference lets us QA both the LLM and the humans, gives a large automatic
agreement metric, and directly answers "you just trusted an LLM." F6 showed the prose sections are
titled, so prose-only gold (responsibilities/requirements) is largely buildable from section titles,
lightening the human load.

## Consequences

- **Leakage guard:** the model input is built from prose (`textSections`/`jsonSections`) with the
  structured label-widgets excluded; fields not recoverable from prose (often salary) are reported
  honestly rather than assumed.
- **Reproducibility:** collect once with the extended fetcher (prose + `dateOfInitialPublicationUtc`),
  freeze the processed train/test JSONL on **HF Hub**; raw HTML is never committed (only
  prose-derived fields leave the machine). Everything downstream replays the frozen dataset; CI runs
  the scorer on `data/fixtures/` only.
- **Split:** temporal by publication date (train = older, test = newest N%), never random.

## S2 resolution (2026-07-28)

Dataset build shipped (`dataset/collect.collect_dataset` → `dataset/build` → `dataset/split` →
`dataset/hf_dataset`, orchestrated by `dataset/run`). Decisions locked here:

- **Split fraction:** `test_fraction = 0.2` (newest 20% → test), in config. The split sorts by
  `(pub_date, offer_id)` for determinism and always leaves a non-empty train side.
- **Usability filters (`build_record`):** drop offers under `min_prose_chars = 200`, without a
  `pub_date` (no temporal key), or with no learnable gold signal. Reposts are deduplicated by offer id
  and by a normalized prose hash, so the same posting can't leak across the train/test boundary.
- **S2 label target = platform-gold** (the structured `__NEXT_DATA__` fields, normalized). The
  triangulated LLM↔platform↔human **agreement QA is S3**, run over this frozen set — not a blocker for
  freezing the dataset.
- **HF push deferred:** `push_dataset` is implemented and gated on an `HF_TOKEN`; the processed JSONL
  is frozen locally (`data/processed/`, gitignored) and uploaded once the dataset repo is provisioned,
  mirroring the deferred GitHub publish. No re-collection is needed to push later.
- Still open for S3: who adjudicates the human-checked gold sample (a strong-LLM first pass
  spot-checked by the human is acceptable, as in S1).

## S3 resolution (2026-07-28)

The two items this ADR left open are decided in **[ADR-0005](0005-labeling-qa-architecture.md)**:
the full-scale proposer is **local Bielik-1.5B few-shot** (a prose-recoverability floor, not a quality
claim), and adjudication is **hybrid with an API arbiter** (`claude-opus-4-8`, distinct from the
proposer) — the human adjudicates every contested cell on a **72-record, disagreement-stratified,
seeded** sample drawn from all 710. The single egress point is ≤80 PII-free prose snippets to the
arbiter, hard-capped in config.

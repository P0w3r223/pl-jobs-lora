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
- Open for S2/S3: the HF dataset repo ID + token (config placeholder today), and who adjudicates the
  probe's 25-example gold (a strong-LLM first pass spot-checked by the human is acceptable for S1).

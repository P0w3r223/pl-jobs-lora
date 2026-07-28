# labeling-QA fixtures

**Synthetic, PII-free** records — the only labeling-QA data committed to git. Real proposals, the
review queue, and human gold echo posting prose and are gitignored (ADR-0005); these five invented
offers exist so the pure agreement layer runs offline in CI and so the on-disk shapes are documented.

- `processed/{train,test}.jsonl` — the frozen-dataset shape: `offer_id, url, pub_date, prose, gold`.
- `proposals.jsonl` — the LLM leg (`fx-002..005`; `fx-001` stands in as an excluded few-shot shot).
- `human_gold.jsonl` — the adjudicated sample (`fx-002, fx-004`), nested `human_gold` form.

`fx-004` is engineered so the LLM and human agree on `work_mode`/`tech_expected` while the platform
gold is empty/short — a *platform-gap-caught* cell, the bucket the triangulation exists to surface.
Exercised by `tests/test_labeling_qa.py::test_build_report_over_committed_fixtures`.

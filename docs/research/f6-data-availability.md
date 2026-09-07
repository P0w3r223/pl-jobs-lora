# F6 — Data-availability gate (theprotocol `__NEXT_DATA__`)

Date: 2026-07-27
Status: accepted (PASS)
Author: Piotr Cząstkiewicz
Related to: docs/decisions/0002-dataset-and-labeling.md

---

## Question

Does one live theprotocol offer's `__NEXT_DATA__` carry (a) the free-text posting **prose**
(the model input) and (b) a **publication date** (for the time-based split)? The it-job-radar
DB stores only structured attributes — its `parse_offer` never captures prose — so if the prose
were not in `__NEXT_DATA__`, the whole P4 approach would need reworking. This gate had to pass
before building anything.

## Method

One live fetch via it-job-radar's own ethical collector (`fetch_sitemap_urls` →
`fetch_offer_html` → `_extract_next_data`), inspecting `props.pageProps.offer`. Read-only, a
single offer. Raw HTML not stored (ADR-0002).

## Verdict: **PASS**

Offer: *Analityk systemowo-biznesowy* (Warszawa). The sitemap lists ~5,834 current offers —
ample headroom for an 800-offer sample.

**Prose (model input) — present and section-titled:**
- `offer.textSections[]` — each with `.title`, `.plainText`, and `.elements[]` (list of strings).
- `offer.jsonSections[]` — `.model.paragraphs[]` / `.model.bullets[]` (structured prose).
- Observed titled sections: *"O projekcie"* (about, 224 chars), *"Twój zakres obowiązków"*
  (responsibilities, 1096 chars), a requirements section (1012 chars), and an offered section.
- Because the sections are **titled**, the prose-only gold fields (responsibilities/requirements)
  can be built largely from section titles — this lightens the labeling QA (see ADR-0002, Q5).

**Publication date (time split) — present:**
- `offer.publicationDetails.dateOfInitialPublicationUtc` (e.g. `2026-07-13T11:32:56Z`), plus
  `lastPublishedUtc` and `archivizationDateUtc` (expiration).

**Labels (platform-gold) — present:**
- `offer.technologies`, `offer.jsonPrimaryAttributes`, and
  `offer.attributes.{employment, specializations, workplaces, title}`.

## Consequence

No data-sourcing rework. S2's extended fetcher captures the prose from `textSections`/`jsonSections`
(excluding the structured widgets that map 1:1 to labels — leakage guard, ADR-0002) plus
`publicationDetails.dateOfInitialPublicationUtc`, and freezes the processed dataset on HF Hub.

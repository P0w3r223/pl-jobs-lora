"""Turn raw collected offers into clean, deduplicated dataset records (pure, offline).

A *record* is the model's train/eval unit: the prose input, the platform-gold JSON target,
and the publication date the temporal split keys on. Building filters out unusable offers
(too little prose, no publication date, no learnable label) and deduplicates reposts, so the
same posting can't inflate the set or leak across the train/test boundary.

Everything here is pure `(examples, knobs) -> (records, stats)` — no network, no filesystem —
so the dataset build is deterministic and unit-testable; I/O lives in ``dataset.run``.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict

from pl_jobs_lora.dataset.collect import DevExample

# Gold fields that carry a learnable signal: an offer whose prose yields none of these is
# noise for a prose -> JSON task (e.g. a page that parsed but had empty attributes).
_SIGNAL_FIELDS = (
    "title", "seniority", "work_mode", "tech_expected", "tech_optional", "salary",
)


def _prose_key(prose: str) -> str:
    """Stable hash of normalized prose — catches reposts of the same offer under a new id."""
    collapsed = " ".join(prose.split()).lower()
    return hashlib.sha1(collapsed.encode("utf-8")).hexdigest()


def _has_signal(gold: dict) -> bool:
    return any(gold.get(f) for f in _SIGNAL_FIELDS)


def build_record(ex: DevExample, *, min_prose_chars: int) -> dict | None:
    """One DevExample -> a JSONL-ready record, or ``None`` if it fails a usability filter.

    Drops offers that are too short to learn from, undatable (no temporal split key), or
    label-empty. The record is a plain dict (``offer_id``, ``url``, ``pub_date``, ``prose``,
    ``gold``) so it serializes directly and stays decoupled from the collector dataclass."""
    if len(ex.prose) < min_prose_chars:
        return None
    if not ex.pub_date:
        return None  # temporal split (ADR-0002) needs a publication date
    if not _has_signal(ex.gold):
        return None
    row = asdict(ex)
    return {k: row[k] for k in ("offer_id", "url", "pub_date", "prose", "gold")}


def dedupe(records: list[dict]) -> list[dict]:
    """Keep the first occurrence of each offer, by id and by prose hash (repost guard)."""
    seen_ids: set[str] = set()
    seen_prose: set[str] = set()
    out: list[dict] = []
    for r in records:
        pkey = _prose_key(r["prose"])
        if r["offer_id"] in seen_ids or pkey in seen_prose:
            continue
        seen_ids.add(r["offer_id"])
        seen_prose.add(pkey)
        out.append(r)
    return out


def build_dataset(
    examples: list[DevExample], *, min_prose_chars: int
) -> tuple[list[dict], dict]:
    """Filter + dedupe collected examples into records, with a stats breakdown for the manifest."""
    kept = [rec for ex in examples if (rec := build_record(ex, min_prose_chars=min_prose_chars))]
    deduped = dedupe(kept)
    stats = {
        "collected": len(examples),
        "passed_filters": len(kept),
        "dropped_filters": len(examples) - len(kept),
        "dropped_duplicates": len(kept) - len(deduped),
        "records": len(deduped),
    }
    return deduped, stats

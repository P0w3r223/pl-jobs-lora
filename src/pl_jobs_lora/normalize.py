"""Normalization: technologies, seniority, work mode, currency.

VENDORED from P0w3r223/it-job-radar @ 2e051cdb (src/it_job_radar/normalize.py), trimmed to
the functions the P4 scorer needs. ADR-0002 chose vendoring over a git dependency for S1:
self-contained and unblocked, at the cost of keeping this copy in sync with the source.

Why reuse rather than reinvent: the harness scores predictions against gold, and both sides
run through *these* functions, so ``ReactJS`` vs ``react`` (alias) and ``regular`` → mid
(the Polish seniority quirk) are matches, not spurious errors — that is what makes the
two-model comparison fair (ADR-0003).

Pure functions — no network, no DB.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from rapidfuzz import fuzz, process

from pl_jobs_lora import vocab

# The vendored alias dictionary lives in the repo (data/normalization/), not in site-packages.
_DEFAULT_ALIASES_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "normalization" / "tech_aliases.yaml"
)


def load_tech_aliases(path: Path | None = None) -> dict[str, str]:
    """Load the alias dictionary as a flat ``alias(lowercase) -> canonical`` index."""
    path = path or _DEFAULT_ALIASES_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    index: dict[str, str] = {}
    for canonical, aliases in raw.items():
        index[canonical.lower()] = canonical
        for alias in aliases or []:
            index[str(alias).lower()] = canonical
    return index


def normalize_technology(
    name: str, alias_index: dict[str, str], threshold: int = vocab.FUZZY_THRESHOLD
) -> str:
    """Map a raw technology name to its canonical form (exact alias → fuzzy → lowercased)."""
    key = (name or "").strip().lower()
    if not key:
        return ""
    if key in alias_index:
        return alias_index[key]
    match = process.extractOne(key, alias_index.keys(), scorer=fuzz.ratio)
    if match and match[1] >= threshold:
        return alias_index[match[0]]
    return key  # unknown technology — keep it, lowercased


def normalize_seniority(value: str | None) -> str | None:
    """Map a seniority label to a canonical level (Polish ``regular`` → mid)."""
    if not value:
        return None
    return vocab.SENIORITY_MAP.get(value.strip().lower(), value.strip().lower())


def normalize_work_mode(code: str | None) -> str | None:
    """Map a work-mode code to remote/hybrid/office/mobile."""
    if not code:
        return None
    return vocab.WORK_MODE_MAP.get(code.strip().lower(), code.strip().lower())


def normalize_currency(code: str | None) -> str | None:
    """Map a currency symbol/code to an ISO code (``zł`` → PLN)."""
    if not code:
        return None
    return vocab.CURRENCY_MAP.get(code.strip().lower(), code.strip().upper())

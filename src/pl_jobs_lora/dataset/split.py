"""Temporal train/test split (pure, offline).

The split is by publication date, never random (ADR-0002): the model is trained on older
postings and evaluated on the newest ones, which is the honest test — new tech and phrasings
appear over time, and a random split would leak that future into training. Deterministic
ordering (date, then offer id) makes the split reproducible across runs and machines.
"""

from __future__ import annotations


def temporal_split(
    records: list[dict], *, test_fraction: float
) -> tuple[list[dict], list[dict]]:
    """Sort by (``pub_date``, ``offer_id``); the newest ``test_fraction`` become test, the rest
    train. Every record must carry a ``pub_date`` (guaranteed by ``build_record``). Guarantees a
    non-empty train side whenever there are >= 2 records, so the split never trains on nothing."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    ordered = sorted(records, key=lambda r: (r["pub_date"], r["offer_id"]))
    n = len(ordered)
    if n < 2:
        return ordered, []  # too few to hold out a test set
    n_test = min(max(1, round(n * test_fraction)), n - 1)
    cut = n - n_test
    return ordered[:cut], ordered[cut:]

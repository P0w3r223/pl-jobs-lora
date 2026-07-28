"""Temporal train/test split: newest -> test, deterministic, never random (ADR-0002)."""

from __future__ import annotations

import pytest

from pl_jobs_lora.dataset.split import temporal_split


def _rec(offer_id: str, pub_date: str) -> dict:
    return {"offer_id": offer_id, "pub_date": pub_date, "prose": "p", "gold": {}}


def _records(n: int) -> list[dict]:
    # Ascending dates d01..dNN, deliberately shuffled to prove the split sorts them.
    recs = [_rec(f"o{i:02d}", f"2026-07-{i:02d}T00:00:00Z") for i in range(1, n + 1)]
    return recs[::-1]


def test_test_side_is_strictly_newer_than_train():
    train, test = temporal_split(_records(10), test_fraction=0.2)
    assert [r["offer_id"] for r in test] == ["o09", "o10"]
    assert max(r["pub_date"] for r in train) < min(r["pub_date"] for r in test)


def test_fraction_rounds_to_nearest():
    train, test = temporal_split(_records(10), test_fraction=0.25)
    assert (len(train), len(test)) == (8, 2)  # round(2.5) -> 2


def test_always_leaves_at_least_one_train_and_one_test():
    train, test = temporal_split(_records(3), test_fraction=0.9)
    assert len(train) == 1 and len(test) == 2  # n_test clamped to n-1


def test_ties_broken_by_offer_id_deterministically():
    recs = [_rec("b", "2026-07-01T00:00:00Z"), _rec("a", "2026-07-01T00:00:00Z")]
    train, test = temporal_split(recs, test_fraction=0.5)
    assert train[0]["offer_id"] == "a" and test[0]["offer_id"] == "b"


def test_single_record_has_no_test():
    train, test = temporal_split([_rec("a", "2026-07-01T00:00:00Z")], test_fraction=0.2)
    assert len(train) == 1 and test == []


def test_empty_input():
    assert temporal_split([], test_fraction=0.2) == ([], [])


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_rejects_out_of_range_fraction(bad):
    with pytest.raises(ValueError):
        temporal_split(_records(5), test_fraction=bad)

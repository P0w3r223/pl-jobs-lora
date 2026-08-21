"""Bootstrap uncertainty (ADR-0003): determinism, interval sanity, and that pairing is real.

Pure and offline — synthetic predictions, no model, no network.
"""

from __future__ import annotations

import pytest

from pl_jobs_lora.eval.bootstrap import bootstrap_variants, render_markdown

_TOL = 0.05
_KNOBS = {"salary_rel_tolerance": _TOL, "resamples": 200, "seed": 7, "ci": 95.0}


def _gold(oid, seniority):
    return {
        "offer_id": oid, "title": None, "seniority": [seniority], "work_mode": [],
        "tech_expected": [], "tech_optional": [], "salary": None,
    }


def _pred(oid, seniority, *, valid=True):
    return {
        "offer_id": oid, "valid": valid,
        "parsed": {"seniority": [seniority]} if valid else None,
    }


# 20 records; "perfect" gets every one right, "half" gets the even-indexed ones right.
GOLD = [_gold(str(i), "mid") for i in range(20)]
PERFECT = [_pred(str(i), "mid") for i in range(20)]
HALF = [_pred(str(i), "mid" if i % 2 == 0 else "senior") for i in range(20)]


def test_is_deterministic_for_a_given_seed():
    a = bootstrap_variants({"v": PERFECT}, GOLD, **_KNOBS).as_dict()
    b = bootstrap_variants({"v": PERFECT}, GOLD, **_KNOBS).as_dict()
    assert a == b


def test_a_different_seed_moves_the_interval():
    a = bootstrap_variants({"v": HALF}, GOLD, **{**_KNOBS, "seed": 1}).as_dict()
    b = bootstrap_variants({"v": HALF}, GOLD, **{**_KNOBS, "seed": 2}).as_dict()
    assert a["intervals"]["v"]["mean_field_f1"]["point"] == \
        b["intervals"]["v"]["mean_field_f1"]["point"]      # the estimate is not resampled
    assert a != b                                          # but the bands are


def test_interval_brackets_the_point_estimate():
    r = bootstrap_variants({"half": HALF}, GOLD, **_KNOBS)
    iv = r.intervals["half"]["mean_field_f1"]
    assert iv.low <= iv.point <= iv.high


def test_a_perfect_variant_has_a_degenerate_interval():
    """Every resample scores 1.0, so there is nothing for the band to span."""
    r = bootstrap_variants({"perfect": PERFECT}, GOLD, **_KNOBS)
    iv = r.intervals["perfect"]["mean_field_f1"]
    assert (iv.point, iv.low, iv.high) == (1.0, 1.0, 1.0)


def test_reference_defaults_to_the_best_variant():
    r = bootstrap_variants({"half": HALF, "perfect": PERFECT}, GOLD, **_KNOBS)
    assert r.reference == "perfect"


def test_a_real_gap_is_separated_from_zero():
    r = bootstrap_variants({"half": HALF, "perfect": PERFECT}, GOLD, **_KNOBS)
    diff = next(p for p in r.paired if p.metric == "mean_field_f1")
    assert diff.variant == "half" and diff.reference == "perfect"
    assert diff.difference.point < 0
    assert diff.separated, "being wrong half the time is not within noise of being perfect"
    assert diff.sign_agreement == 1.0


def test_two_identical_variants_are_not_separated():
    """The property that matters: no gap must not be reported as a gap."""
    r = bootstrap_variants({"a": HALF, "b": list(HALF)}, GOLD, **_KNOBS)
    for p in r.paired:
        assert p.difference.point == 0.0
        assert not p.separated
        assert p.sign_agreement == 1.0, "a dead heat ties on every resample"


def test_pairing_cancels_the_shared_draw():
    """Identical variants must differ by exactly zero on *every* resample, not just on average.

    This is what distinguishes paired from independent resampling: independently drawn samples
    would give two different-but-similar scores per iteration, so the difference would wobble
    around zero instead of being identically zero.
    """
    r = bootstrap_variants({"a": HALF, "b": list(HALF)}, GOLD, **_KNOBS)
    diff = next(p for p in r.paired if p.metric == "mean_field_f1")
    assert (diff.difference.low, diff.difference.high) == (0.0, 0.0)


def test_a_partial_variant_is_not_compared_as_if_it_were_paired():
    """Pairing only cancels the draw when both sides were scored on the same records.

    A half-coverage variant is scored on its own subset of every resample, so its difference from
    a complete variant is between two populations. Reported as not comparable rather than as a
    tie — otherwise a run that answered half the set reads as indistinguishable from a full one.
    """
    partial = PERFECT[:10]
    r = bootstrap_variants({"partial": partial, "perfect": PERFECT}, GOLD, **_KNOBS)
    assert r.coverage["partial"] == 0.5 and r.coverage["perfect"] == 1.0
    for p in r.paired:
        assert p.comparable is False
        assert p.separated is False, "an unpaired difference is never a finding"
    assert "n/a" in render_markdown(r)
    assert "answered only part of the gold set" in render_markdown(r)


def test_full_coverage_pairs_stay_comparable():
    r = bootstrap_variants({"half": HALF, "perfect": PERFECT}, GOLD, **_KNOBS)
    assert all(p.comparable for p in r.paired)


def test_duplicate_rows_are_scored_as_the_main_table_scores_them():
    """The rendered text calls the point estimates "the numbers in the table above" — so they must
    be. Deduping by offer_id here while the scorer iterates rows made the two disagree."""
    from pl_jobs_lora.eval.scoring import mean_measured_f1, score_predictions

    duped = [*HALF, HALF[1]]                       # one record answered twice
    scored = score_predictions(duped, GOLD, salary_rel_tolerance=_TOL).as_dict()
    r = bootstrap_variants({"duped": duped}, GOLD, **_KNOBS)
    assert r.intervals["duped"]["mean_field_f1"].point == mean_measured_f1(scored["fields"])
    assert r.intervals["duped"]["json_validity"].point == scored["json_validity"]


def test_predictions_outside_the_gold_set_are_ignored():
    stray = [*PERFECT, _pred("not-in-gold", "mid")]
    r = bootstrap_variants({"stray": stray}, GOLD, **_KNOBS)
    assert r.coverage["stray"] == 1.0, "coverage counts answered gold records, not stray rows"


def test_invalid_predictions_lower_the_validity_interval():
    broken = [_pred(str(i), "mid", valid=(i % 2 == 0)) for i in range(20)]
    r = bootstrap_variants({"broken": broken}, GOLD, **_KNOBS)
    iv = r.intervals["broken"]["json_validity"]
    assert iv.point == 0.5
    assert 0.0 < iv.low < 0.5 < iv.high < 1.0


def test_render_reports_the_seed_and_both_sections():
    r = bootstrap_variants({"half": HALF, "perfect": PERFECT}, GOLD, **_KNOBS)
    md = render_markdown(r)
    assert "## Uncertainty (bootstrap)" in md
    assert "seed 7" in md and "200 resamples" in md
    assert "Paired differences vs `perfect`" in md


@pytest.mark.parametrize(("knob", "value"), [("resamples", 0), ("ci", 0.0), ("ci", 100.0)])
def test_rejects_nonsensical_knobs(knob, value):
    with pytest.raises(ValueError):
        bootstrap_variants({"v": PERFECT}, GOLD, **{**_KNOBS, knob: value})


def test_rejects_an_empty_gold_set():
    with pytest.raises(ValueError):
        bootstrap_variants({"v": []}, [], **_KNOBS)

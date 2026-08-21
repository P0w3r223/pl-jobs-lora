"""Pure scorer: perfect match, invalid-JSON penalty, salary tolerance, and silence earning zero."""

from __future__ import annotations

import pytest

from pl_jobs_lora.eval.scoring import (
    HEADLINE_FIELDS,
    SET_FIELDS,
    mean_measured_f1,
    score_predictions,
)

_GOLD = [{
    "offer_id": "a",
    "title": "Senior Python Developer",
    "seniority": ["senior", "mid"], "work_mode": ["remote"],
    "tech_expected": ["python", "sql"], "tech_optional": ["react"],
    "salary": {"kind": "b2b", "currency": "PLN", "amount_from": 100,
               "amount_to": 140, "period": "hour"},
}]

# The dominant shape of the real test set: ~69 % of gold records carry no salary at all.
_GOLD_NO_SALARY = [dict(_GOLD[0], salary=None)]


def _pred(parsed, valid=True):
    return [{"offer_id": "a", "valid": valid, "parsed": parsed}]


def _score(preds, gold=_GOLD):
    return score_predictions(preds, gold, salary_rel_tolerance=0.05).as_dict()


def test_perfect_match_scores_one():
    r = _score(_pred(dict(_GOLD[0])))
    assert r["json_validity"] == 1.0
    assert r["title_exact"] == 1.0
    assert r["fields"]["tech_expected"]["f1"] == 1.0
    assert r["salary"]["detection"] == 1.0
    assert (r["salary"]["currency"], r["salary"]["kind"], r["salary"]["amount"]) == (1.0, 1.0, 1.0)


def test_invalid_json_scores_zero_everywhere():
    r = _score(_pred(None, valid=False))
    assert r["json_validity"] == 0.0
    assert r["fields"]["tech_expected"]["f1"] == 0.0
    assert r["salary"]["amount"] == 0.0


def test_invalid_prediction_earns_nothing_when_gold_salary_is_absent():
    """The regression this scorer previously failed.

    With ``None == None`` credited, an unparseable prediction scored 1.0 on currency/kind/amount
    for every record without a salary — and ~69 % of gold has none, which is how a model at
    0.049 JSON validity reached ~0.75 on all three salary sub-metrics.
    """
    r = _score(_pred(None, valid=False), _GOLD_NO_SALARY)
    salary = r["salary"]
    assert salary["detection"] == 0.0, "an unparseable prediction decided nothing"
    assert salary["support"] == 0
    assert salary["currency"] is None and salary["kind"] is None and salary["amount"] is None


def test_correctly_predicting_no_salary_scores_detection_but_no_value_credit():
    """Saying 'absent' correctly is a real answer — but it is detection, not value accuracy."""
    r = _score(_pred(dict(_GOLD[0], salary=None)), _GOLD_NO_SALARY)
    assert r["salary"]["detection"] == 1.0
    assert r["salary"]["support"] == 0
    assert r["salary"]["currency"] is None  # nothing to be accurate about


def test_missing_a_salary_that_gold_has_is_a_detection_failure():
    r = _score(_pred(dict(_GOLD[0], salary=None)))
    assert r["salary"]["detection"] == 0.0
    assert r["salary"]["support"] == 1
    assert r["salary"]["amount"] == 0.0


def test_salary_amount_within_tolerance():
    parsed = dict(_GOLD[0], salary=dict(_GOLD[0]["salary"], amount_from=103, amount_to=138))
    assert _score(_pred(parsed))["salary"]["amount"] == 1.0  # 103 ~ 100, 138 ~ 140 within 5 %


def test_salary_amount_outside_tolerance():
    parsed = dict(_GOLD[0], salary=dict(_GOLD[0]["salary"], amount_from=200))
    assert _score(_pred(parsed))["salary"]["amount"] == 0.0


def test_partial_tech_recall():
    parsed = dict(_GOLD[0], tech_expected=["python"])  # missing sql
    f = _score(_pred(parsed))["fields"]["tech_expected"]
    assert f["precision"] == 1.0 and f["recall"] == 0.5


def test_empty_field_on_both_sides_is_unmeasurable_not_perfect():
    """Empty-vs-empty is not an achievement: no support, so the field reports None, not 1.0."""
    gold = [dict(_GOLD[0], tech_optional=[])]
    parsed = dict(_GOLD[0], tech_optional=[])
    f = _score(_pred(parsed), gold)["fields"]["tech_optional"]
    assert f["f1"] is None
    assert f["support"] == 0
    assert f["exact_match"] is None


def test_predicting_nothing_does_not_earn_exact_match():
    gold = [dict(_GOLD[0], tech_optional=["react"])]
    parsed = dict(_GOLD[0], tech_optional=[])
    f = _score(_pred(parsed), gold)["fields"]["tech_optional"]
    assert f["exact_match"] == 0.0
    assert f["support"] == 1
    assert f["f1"] == 0.0


def test_headline_excludes_tech_optional_but_still_scores_it():
    """Reported because it is in the gold; not in the headline because it is not in the prose.

    Pinned because this moves a published number: dropping a field that scores ~0 for every
    variant raises every headline. The exclusion must be a stated decision, not a silent one.
    """
    assert "tech_optional" in SET_FIELDS, "still scored and reported"
    assert "tech_optional" not in HEADLINE_FIELDS, "not averaged into the headline"

    fields = {
        "seniority": {"f1": 0.6}, "work_mode": {"f1": 0.6},
        "tech_expected": {"f1": 0.3}, "tech_optional": {"f1": 0.0},
    }
    assert mean_measured_f1(fields) == pytest.approx(0.5)
    assert mean_measured_f1(fields, headline_fields=SET_FIELDS) == pytest.approx(0.375)


def test_headline_skips_unmeasurable_fields_without_counting_them_as_zero():
    fields = {"seniority": {"f1": 0.6}, "work_mode": {"f1": None}, "tech_expected": {"f1": 0.4}}
    assert mean_measured_f1(fields) == pytest.approx(0.5)
    assert mean_measured_f1({"seniority": {"f1": None}}) is None


def test_coverage_exposes_a_partial_run():
    gold = [dict(_GOLD[0], offer_id=str(i)) for i in range(10)]
    preds = [{"offer_id": "0", "valid": True, "parsed": dict(_GOLD[0])}]
    r = score_predictions(preds, gold, salary_rel_tolerance=0.05).as_dict()
    assert r["n"] == 1 and r["n_gold"] == 10
    assert r["coverage"] == 0.1, "a run that died at record 1 of 10 must not read as complete"


def test_coverage_counts_distinct_records_not_prediction_rows():
    """A duplicated prediction must not buy coverage of a gold set it never answered.

    `coverage` is presented as a completeness guarantee, so the one way it could lie — an
    appended-twice predictions file, which the labeling-QA resume path writes — is pinned here.
    """
    gold = [dict(_GOLD[0], offer_id=str(i)) for i in range(10)]
    preds = [{"offer_id": "0", "valid": True, "parsed": dict(_GOLD[0])}] * 5
    r = score_predictions(preds, gold, salary_rel_tolerance=0.05).as_dict()
    assert r["coverage"] == 0.1, "five copies of one answer still answer one record of ten"
    assert r["n"] == 5, "n stays the row count — coverage is what carries the completeness claim"
    assert r["n_duplicate_rows"] == 4


def test_duplicates_at_full_coverage_are_counted_not_silent():
    """The last way these metrics could mislead: full coverage that double-weights a record.

    `coverage` alone reads 1.00 here, so the completeness claim looks clean while the repeated
    record counts twice in tp/fp/fn. The duplicate count is what makes that visible.
    """
    gold = [dict(_GOLD[0], offer_id=str(i)) for i in range(3)]
    preds = [
        {"offer_id": str(i), "valid": True, "parsed": dict(_GOLD[0])} for i in (0, 1, 2, 2)
    ]
    r = score_predictions(preds, gold, salary_rel_tolerance=0.05).as_dict()
    assert r["coverage"] == 1.0
    assert r["n"] == 4 and r["n_duplicate_rows"] == 1

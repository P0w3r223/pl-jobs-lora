"""Pure scorer: perfect match, invalid-JSON penalty, salary tolerance, empty-field vacuity."""

from __future__ import annotations

from pl_jobs_lora.eval.scoring import score_predictions

_GOLD = [{
    "offer_id": "a",
    "title": "Senior Python Developer",
    "seniority": ["senior", "mid"], "work_mode": ["remote"],
    "tech_expected": ["python", "sql"], "tech_optional": ["react"],
    "salary": {"kind": "b2b", "currency": "PLN", "amount_from": 100,
               "amount_to": 140, "period": "hour"},
}]


def _pred(parsed, valid=True):
    return [{"offer_id": "a", "valid": valid, "parsed": parsed}]


def test_perfect_match_scores_one():
    r = score_predictions(_pred(dict(_GOLD[0])), _GOLD, salary_rel_tolerance=0.05).as_dict()
    assert r["json_validity"] == 1.0
    assert r["title_exact"] == 1.0
    assert r["fields"]["tech_expected"]["f1"] == 1.0
    assert r["salary"] == {"currency": 1.0, "kind": 1.0, "amount": 1.0}


def test_invalid_json_scores_zero_everywhere():
    r = score_predictions(_pred(None, valid=False), _GOLD, salary_rel_tolerance=0.05).as_dict()
    assert r["json_validity"] == 0.0
    assert r["fields"]["tech_expected"]["f1"] == 0.0
    assert r["salary"]["amount"] == 0.0


def test_salary_amount_within_tolerance():
    parsed = dict(_GOLD[0], salary=dict(_GOLD[0]["salary"], amount_from=103, amount_to=138))
    r = score_predictions(_pred(parsed), _GOLD, salary_rel_tolerance=0.05).as_dict()
    assert r["salary"]["amount"] == 1.0  # 103 within 5% of 100, 138 within 5% of 140


def test_salary_amount_outside_tolerance():
    parsed = dict(_GOLD[0], salary=dict(_GOLD[0]["salary"], amount_from=200))
    r = score_predictions(_pred(parsed), _GOLD, salary_rel_tolerance=0.05).as_dict()
    assert r["salary"]["amount"] == 0.0


def test_partial_tech_recall():
    parsed = dict(_GOLD[0], tech_expected=["python"])  # missing sql
    r = score_predictions(_pred(parsed), _GOLD, salary_rel_tolerance=0.05).as_dict()
    f = r["fields"]["tech_expected"]
    assert f["precision"] == 1.0 and f["recall"] == 0.5


def test_empty_field_both_sides_is_vacuous_match():
    gold = [dict(_GOLD[0], tech_optional=[])]
    parsed = dict(_GOLD[0], tech_optional=[])
    r = score_predictions(_pred(parsed), gold, salary_rel_tolerance=0.05).as_dict()
    assert r["fields"]["tech_optional"]["f1"] == 1.0

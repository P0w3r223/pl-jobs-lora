"""Output parser: JSON extraction, normalization before validation, invalid-output handling."""

from __future__ import annotations

import json

import pytest

from pl_jobs_lora.eval import scoring
from pl_jobs_lora.eval.prompt import parse_output, parse_result
from pl_jobs_lora.normalize import load_tech_aliases

_ALIASES = load_tech_aliases()


def test_parses_and_normalizes():
    raw = 'Here you go: ' + json.dumps({
        "title": "Backend Engineer",
        "seniority": ["regular"], "work_mode": ["home-office"],
        "tech_expected": ["ReactJS"], "tech_optional": [],
        "salary": {"kind": "b2b", "currency": "zł", "amount_from": 100, "amount_to": 140},
    })
    parsed, valid = parse_output(raw, _ALIASES)
    assert valid
    assert parsed["seniority"] == ["mid"]        # regular -> mid
    assert parsed["work_mode"] == ["remote"]     # home-office -> remote
    assert parsed["tech_expected"] == ["react"]  # ReactJS -> react
    assert parsed["salary"]["currency"] == "PLN"


def test_hallucinated_key_is_invalid():
    raw = json.dumps({"title": "X", "made_up_field": 1})
    parsed, valid = parse_output(raw, _ALIASES)
    assert parsed is None and valid is False  # extra=forbid


def test_non_json_is_invalid():
    parsed, valid = parse_output("I cannot help with that.", _ALIASES)
    assert parsed is None and valid is False


def test_trailing_garbage_after_object_is_tolerated():
    parsed, valid = parse_output('{"title": "X"} <eos> blah', _ALIASES)
    assert valid and parsed["title"] == "X"


def test_malformed_salary_currency_type_does_not_crash():
    raw = json.dumps({"title": "X", "salary": {"currency": ["PLN", "EUR"]}})
    parsed, valid = parse_output(raw, _ALIASES)
    assert valid
    assert parsed["salary"]["currency"] is None


def test_scalar_instead_of_list_field_does_not_crash():
    raw = json.dumps({"title": "X", "seniority": 5, "work_mode": 3.0, "tech_expected": 7})
    parsed, valid = parse_output(raw, _ALIASES)
    assert valid
    assert parsed["seniority"] == []
    assert parsed["work_mode"] == []
    assert parsed["tech_expected"] == []


def test_string_instead_of_list_field_is_not_split_into_characters():
    raw = json.dumps({"title": "X", "seniority": "mid", "tech_expected": "react"})
    parsed, valid = parse_output(raw, _ALIASES)
    assert valid
    assert parsed["seniority"] == []
    assert parsed["tech_expected"] == []


# --- failure taxonomy: `valid` says how often, `failure` says because of what ------------------

@pytest.mark.parametrize(("raw", "expected"), [
    ("", scoring.EMPTY_OUTPUT),
    ("   \n\t ", scoring.EMPTY_OUTPUT),
    ("I cannot help with that.", scoring.NO_JSON_OBJECT),
    ('{"title": "X", "seniority": [', scoring.JSON_DECODE_ERROR),   # the truncation shape
    ('{"title": "X", "made_up_field": 1}', scoring.SCHEMA_INVALID),
])
def test_each_failure_class_is_distinguished(raw, expected):
    result = parse_result(raw, _ALIASES)
    assert result.valid is False and result.parsed is None
    assert result.failure == expected


def test_an_object_wrapped_in_a_list_still_parses():
    """Why the taxonomy has no "not an object" class: decoding anchors on the first `{`."""
    result = parse_result('[{"title": "X"}]', _ALIASES)
    assert result.valid and result.parsed["title"] == "X"


def test_a_valid_parse_records_no_failure():
    result = parse_result(json.dumps({"title": "X"}), _ALIASES)
    assert result.valid is True and result.failure is None


def test_every_failure_class_is_declared_in_the_taxonomy():
    """A class the parser can emit but the report does not know would vanish from the table."""
    emitted = {
        parse_result(raw, _ALIASES).failure
        for raw in ("", "prose only", '{"a": ', '{"made_up_field": 1}')
    }
    assert emitted <= set(scoring.PARSE_FAILURES)


def test_parse_output_tuple_form_still_agrees_with_the_primitive():
    """The labeling-QA arbiter still consumes the tuple; the two must not drift."""
    raw = json.dumps({"title": "X", "seniority": ["regular"]})
    result = parse_result(raw, _ALIASES)
    assert parse_output(raw, _ALIASES) == (result.parsed, result.valid)

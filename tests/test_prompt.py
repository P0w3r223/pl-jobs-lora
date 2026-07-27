"""Output parser: JSON extraction, normalization before validation, invalid-output handling."""

from __future__ import annotations

import json

from pl_jobs_lora.eval.prompt import parse_output
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

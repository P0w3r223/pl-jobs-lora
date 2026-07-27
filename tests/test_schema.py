"""The extraction contract round-trips valid postings and rejects malformed ones."""

import pytest
from pydantic import ValidationError

from pl_jobs_lora.schema import JobPosting, SalaryKind, Seniority, WorkMode

VALID = {
    "title": "Analityk systemowo-biznesowy",
    "seniority": ["mid", "senior"],
    "work_mode": ["remote", "hybrid"],
    "tech_expected": ["Python", "SQL"],
    "tech_optional": ["Docker"],
    "responsibilities": ["Prowadzenie analiz systemowych."],
    "requirements": ["Minimum 4 lata doświadczenia."],
    "contract_types": ["b2b", "employment"],
    "salary": {"kind": "b2b", "currency": "PLN", "amount_from": 18000, "amount_to": 24000,
               "period": "month"},
}


def test_valid_posting_round_trips():
    posting = JobPosting.model_validate(VALID)
    assert posting.title == "Analityk systemowo-biznesowy"
    assert Seniority.senior in posting.seniority
    assert WorkMode.remote in posting.work_mode
    assert posting.salary is not None and posting.salary.kind is SalaryKind.b2b
    # dump → load is stable
    assert JobPosting.model_validate(posting.model_dump()) == posting


def test_empty_posting_is_valid_with_defaults():
    posting = JobPosting.model_validate({})
    assert posting.seniority == [] and posting.tech_expected == [] and posting.salary is None


def test_hallucinated_extra_field_is_rejected():
    with pytest.raises(ValidationError):
        JobPosting.model_validate({**VALID, "salary_confidence": 0.9})


def test_non_canonical_seniority_is_rejected():
    # "regular" must be normalized to "mid" BEFORE validation, not accepted raw.
    with pytest.raises(ValidationError):
        JobPosting.model_validate({**VALID, "seniority": ["regular"]})


def test_prompt_schema_lists_the_fields():
    schema = JobPosting.prompt_schema()
    assert "seniority" in schema and "tech_expected" in schema and "salary" in schema

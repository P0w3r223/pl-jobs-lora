"""The extraction target contract — Polish IT job-posting prose → JSON.

This is the single source of truth for what the model must produce and what the harness
scores. The categorical fields are enums drawn from the canonical vocabulary (``vocab``)
so the schema itself documents the allowed values in its JSON Schema — which is what we
put in the prompt (``JobPosting.prompt_schema()``). Raw model output is normalized to
this vocabulary *before* validation (see the probe's parser), so a model saying
``"regular"`` is mapped to ``mid`` rather than rejected.

Frozen-ish value object: extra keys are forbidden so a hallucinated field fails validation
(that failure is a first-class JSON-validity signal in the eval, ADR-0003).
"""

from __future__ import annotations

import json
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from pl_jobs_lora import vocab

# Enums built from the canonical vocabulary so there is exactly one list of allowed values.
Seniority = Enum("Seniority", {v: v for v in vocab.SENIORITY_ORDER}, type=str)
WorkMode = Enum("WorkMode", {v: v for v in vocab.WORK_MODE_CANON}, type=str)
SalaryKind = Enum(
    "SalaryKind",
    {vocab.CONTRACT_B2B: vocab.CONTRACT_B2B, vocab.CONTRACT_EMPLOYMENT: vocab.CONTRACT_EMPLOYMENT},
    type=str,
)


class Salary(BaseModel):
    """A salary range. B2B (net) and employment (gross) are never conflated — the ``kind``
    field keeps them apart, matching it-job-radar's salary model."""

    model_config = ConfigDict(extra="forbid")

    kind: SalaryKind | None = None
    currency: str | None = None
    amount_from: float | None = None
    amount_to: float | None = None
    period: str | None = None  # "month" | "hour"


class JobPosting(BaseModel):
    """Structured extraction of one Polish IT job posting."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    seniority: list[Seniority] = Field(default_factory=list)
    work_mode: list[WorkMode] = Field(default_factory=list)
    tech_expected: list[str] = Field(default_factory=list)
    tech_optional: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    contract_types: list[str] = Field(default_factory=list)
    salary: Salary | None = None

    @classmethod
    def prompt_schema(cls) -> str:
        """A compact JSON Schema string to embed in the extraction prompt, so both API
        baselines and the fine-tuned model are asked for the exact same shape."""
        return json.dumps(cls.model_json_schema(), ensure_ascii=False, indent=2)

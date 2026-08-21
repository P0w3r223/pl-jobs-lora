"""Extraction prompt + output parser — shared by the probe and (S4) API baselines.

The prompt is identical across variants (ADR-0003 fairness): only the model differs. The parser
extracts the first JSON object from the raw output and normalizes categorical fields through the
vendored maps *before* schema validation, so ``regular`` -> mid and ``ReactJS`` -> react are
accepted, while a hallucinated key still fails validation (a JSON-validity signal).

A failed parse also records **why** it failed, using the taxonomy in :mod:`pl_jobs_lora.eval.
scoring` (``PARSE_FAILURES``). ``valid`` alone answers "how often", never "because of what": a run
at 0.05 validity is indistinguishable from one that emitted no JSON at all and one that emitted
JSON the schema rejected — and re-deriving the difference means paying for the run again. The
classes run from "produced nothing" to "produced the right shape with wrong contents", so their
distribution reads as a diagnosis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import ValidationError

from pl_jobs_lora import normalize, vocab
from pl_jobs_lora.dataset.collect import DevExample
from pl_jobs_lora.eval.scoring import (
    EMPTY_OUTPUT,
    JSON_DECODE_ERROR,
    NO_JSON_OBJECT,
    SCHEMA_INVALID,
)
from pl_jobs_lora.schema import JobPosting


@dataclass(frozen=True)
class ParseResult:
    """One parsed prediction: the normalized record, whether it is usable, and why not."""

    parsed: dict | None
    valid: bool
    failure: str | None   # None exactly when valid; otherwise one of PARSE_FAILURES

_SYSTEM = (
    "You extract structured data from Polish IT job postings. "
    "Return ONLY one JSON object matching the schema. No prose, no markdown fences."
)

_INSTRUCTION = (
    "Extract the fields below from the posting. Use null / empty lists when a field is absent "
    "in the text. Do not invent values.\n\nJSON schema:\n{schema}"
)


def build_messages(
    example: DevExample, few_shot: list[DevExample], *, n_shots: int
) -> list[dict]:
    """Chat messages for one offer; ``n_shots`` gold examples precede the target (0 = zero-shot)."""
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _INSTRUCTION.format(schema=JobPosting.prompt_schema())},
        {"role": "assistant", "content": "Understood. Send the posting."},
    ]
    for shot in few_shot[:n_shots]:
        messages.append({"role": "user", "content": f"Posting:\n{shot.prose}"})
        messages.append(
            {"role": "assistant", "content": json.dumps(shot.gold, ensure_ascii=False)}
        )
    messages.append({"role": "user", "content": f"Posting:\n{example.prose}"})
    return messages


def _extract_json(raw: str) -> tuple[dict | None, str | None]:
    """First JSON object in ``raw`` -> (object, None), or (None, failure class).

    Decoding is anchored on the first ``{``, so a success is always a dict — an object wrapped in
    a list still parses, and there is no "decoded something that isn't an object" outcome to
    report. Trailing text after the object is ignored (models append prose and stop tokens).
    """
    if not raw.strip():
        return None, EMPTY_OUTPUT
    start = raw.find("{")
    if start < 0:
        return None, NO_JSON_OBJECT
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw[start:])
    except (json.JSONDecodeError, RecursionError):
        # RecursionError: deeply nested output blows the decoder's stack. A model can emit that,
        # and crashing here would abandon a run whose API calls are already paid for — it is one
        # more way the output is unusable, not an exceptional condition.
        return None, JSON_DECODE_ERROR
    return obj, None


def _norm_set(values, mapper, allowed: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for v in values if isinstance(values, list) else []:
        c = mapper(v if isinstance(v, str) else str(v))
        if c in allowed and c not in out:
            out.append(c)
    return out


def parse_result(raw: str, alias_index: dict[str, str]) -> ParseResult:
    """Raw model text -> normalized record + validity + failure class (the full parse primitive)."""
    obj, failure = _extract_json(raw)
    if obj is None:
        return ParseResult(parsed=None, valid=False, failure=failure)

    obj["seniority"] = _norm_set(
        obj.get("seniority"), normalize.normalize_seniority, vocab.SENIORITY_ORDER
    )
    obj["work_mode"] = _norm_set(
        obj.get("work_mode"), normalize.normalize_work_mode, vocab.WORK_MODE_CANON
    )
    for f in ("tech_expected", "tech_optional"):
        values = obj.get(f)
        obj[f] = [
            normalize.normalize_technology(t, alias_index)
            for t in (values if isinstance(values, list) else [])
            if isinstance(t, str) and t.strip()
        ]
    salary = obj.get("salary")
    if isinstance(salary, dict):
        currency = salary.get("currency")
        salary["currency"] = normalize.normalize_currency(
            currency if isinstance(currency, str) else None
        )

    try:
        record = JobPosting.model_validate(obj).model_dump()
    except ValidationError:
        return ParseResult(parsed=None, valid=False, failure=SCHEMA_INVALID)
    return ParseResult(parsed=record, valid=True, failure=None)


def parse_output(raw: str, alias_index: dict[str, str]) -> tuple[dict | None, bool]:
    """Raw model text -> (normalized JobPosting dict, valid); unparseable -> (None, False).

    The tuple form kept for callers that only decide usability (the labeling-QA arbiter). Anything
    writing a predictions file should use :func:`parse_result` and persist ``failure`` too.
    """
    result = parse_result(raw, alias_index)
    return result.parsed, result.valid

"""Extraction prompt + output parser — shared by the probe and (S4) API baselines.

The prompt is identical across variants (ADR-0003 fairness): only the model differs. The parser
extracts the first JSON object from the raw output and normalizes categorical fields through the
vendored maps *before* schema validation, so ``regular`` -> mid and ``ReactJS`` -> react are
accepted, while a hallucinated key still fails validation (a JSON-validity signal).
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from pl_jobs_lora import normalize, vocab
from pl_jobs_lora.dataset.collect import DevExample
from pl_jobs_lora.schema import JobPosting

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


def _extract_json(raw: str) -> dict | None:
    start = raw.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw[start:])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _norm_set(values, mapper, allowed: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for v in values if isinstance(values, list) else []:
        c = mapper(v if isinstance(v, str) else str(v))
        if c in allowed and c not in out:
            out.append(c)
    return out


def parse_output(raw: str, alias_index: dict[str, str]) -> tuple[dict | None, bool]:
    """Raw model text -> (normalized JobPosting dict, valid); unparseable -> (None, False)."""
    obj = _extract_json(raw)
    if obj is None:
        return None, False

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
        return JobPosting.model_validate(obj).model_dump(), True
    except ValidationError:
        return None, False

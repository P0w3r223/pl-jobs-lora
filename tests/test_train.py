"""Pure QLoRA prep (ADR-0006): SFT formatting, temporal dev split, completion masking/encoding.

Offline — no model, no tokenizer (a fake tokenizer exercises the encode/truncation logic).
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from pl_jobs_lora.train.qlora import (
    completion_labels,
    encode_example,
    temporal_dev_split,
    to_sft_example,
)


def _gold(**over):
    base = {
        "title": None, "seniority": [], "work_mode": [], "tech_expected": [],
        "tech_optional": [], "responsibilities": [], "requirements": [],
        "contract_types": [], "salary": None,
    }
    return {**base, **over}


def _record(offer_id, *, pub_date=None, **gold_over):
    return {
        "offer_id": offer_id, "url": f"https://x/{offer_id}", "pub_date": pub_date,
        "prose": f"prose for {offer_id}", "gold": _gold(**gold_over),
    }


# -- SFT formatting --------------------------------------------------------------------------------

def test_to_sft_example_zero_shot_prompt_and_gold_target():
    sft = to_sft_example(_record("a", title="Dev", seniority=["mid"]))
    # zero-shot: the fixed 3-message preamble then the single posting turn — no exemplar pairs
    assert sft["prompt_messages"][0]["role"] == "system"
    assert sft["prompt_messages"][-1]["content"].endswith("prose for a")
    completion = json.loads(sft["completion"])
    assert completion["title"] == "Dev" and completion["seniority"] == ["mid"]


def test_to_sft_example_rejects_malformed_gold():
    rec = _record("a")
    rec["gold"]["seniority"] = ["not-a-real-seniority"]  # outside the vocab enum
    with pytest.raises(ValidationError):
        to_sft_example(rec)


# -- temporal dev split ----------------------------------------------------------------------------

def test_temporal_dev_split_takes_the_newest():
    recs = [_record(f"r{i}", pub_date=f"2026-01-{i + 1:02d}T00:00:00Z") for i in range(10)]
    train, dev = temporal_dev_split(recs, 0.2)
    assert len(dev) == 2 and len(train) == 8
    assert {r["offer_id"] for r in dev} == {"r8", "r9"}  # newest two
    assert {r["offer_id"] for r in dev}.isdisjoint({r["offer_id"] for r in train})


def test_temporal_dev_split_min_one_and_deterministic():
    recs = [_record(f"r{i}", pub_date=f"2026-01-{i + 1:02d}T00:00:00Z") for i in range(5)]
    a = temporal_dev_split(recs, 0.01)  # rounds to 0 -> floored to 1
    b = temporal_dev_split(recs, 0.01)
    assert len(a[1]) == 1 and a[1][0]["offer_id"] == "r4"
    assert [r["offer_id"] for r in a[0]] == [r["offer_id"] for r in b[0]]


# -- masking + encoding ----------------------------------------------------------------------------

def test_completion_labels_masks_the_prompt():
    assert completion_labels(3, [10, 11, 12, 20, 21]) == [-100, -100, -100, 20, 21]


class _FakeTok:
    """Deterministic chat template: prompt -> [1..5]; full (with completion) -> [1..5, 6, 7]."""

    def apply_chat_template(self, msgs, add_generation_prompt=False, **kw):
        return [1, 2, 3, 4, 5] if add_generation_prompt else [1, 2, 3, 4, 5, 6, 7]


_SFT = {"prompt_messages": [{"role": "user", "content": "x"}], "completion": "{}"}


def test_encode_example_no_truncation_masks_completion_only():
    enc = encode_example(_FakeTok(), _SFT, max_seq_len=10)
    assert enc["input_ids"] == [1, 2, 3, 4, 5, 6, 7]
    assert enc["labels"] == [-100, -100, -100, -100, -100, 6, 7]
    assert enc["attention_mask"] == [1] * 7


def test_encode_example_overflow_trims_prompt_keeps_completion():
    enc = encode_example(_FakeTok(), _SFT, max_seq_len=6)
    # completion [6,7] kept whole; budget 4 -> prompt trimmed to [1,2,3,4]
    assert enc["input_ids"] == [1, 2, 3, 4, 6, 7]
    assert enc["labels"] == [-100, -100, -100, -100, 6, 7]

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


def test_temporal_dev_split_keeps_at_least_one_train_record():
    recs = [_record(f"r{i}", pub_date=f"2026-01-{i + 1:02d}T00:00:00Z") for i in range(5)]
    train, dev = temporal_dev_split(recs, 0.99)  # would round to all-dev without the clamp
    assert len(train) >= 1 and len(train) + len(dev) == 5


# -- masking + encoding ----------------------------------------------------------------------------

def test_completion_labels_masks_the_prompt():
    assert completion_labels(3, [10, 11, 12, 20, 21]) == [-100, -100, -100, 20, 21]


class _FakeTok:
    """ChatML-ish content-sensitive template: one token per whitespace word across messages, with an
    assistant-header token inserted before any completion — present in BOTH the prompt and the full
    sequence, so the prompt is a true token prefix (the invariant ``encode_example`` asserts). ids
    are positional, so shortening a message's text reduces the token count."""

    def apply_chat_template(self, msgs, add_generation_prompt=False, **kw):
        pre, comp = [], []
        for m in msgs:
            (comp if m["role"] == "assistant" else pre).extend(m["content"].split())
        seq = [*pre, "<assist>", *comp]  # header sits between the prompt scaffold and completion
        return list(range(len(seq)))


def _sft(posting_words: int):
    return {
        "prompt_messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "Posting: " + " ".join(["w"] * posting_words)},
        ],
        "completion": "{}",  # one token
    }


def test_encode_example_no_truncation_masks_completion_only():
    enc = encode_example(_FakeTok(), _sft(3), max_seq_len=50)
    # completion is the single trailing token; everything before it is masked
    assert enc["labels"][-1] == enc["input_ids"][-1]
    assert enc["labels"].count(-100) == len(enc["input_ids"]) - 1
    assert enc["attention_mask"] == [1] * len(enc["input_ids"])


def test_encode_example_overflow_shrinks_posting_keeps_completion_and_header():
    enc = encode_example(_FakeTok(), _sft(40), max_seq_len=8)
    assert len(enc["input_ids"]) <= 8                       # shrunk to fit
    assert enc["labels"][-1] == enc["input_ids"][-1]        # completion still trained
    assert enc["labels"].count(-100) == len(enc["input_ids"]) - 1
    assert enc["labels"].count(-100) >= 2                   # system + header scaffold kept

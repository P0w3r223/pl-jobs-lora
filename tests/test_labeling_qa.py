"""Labeling-QA pure parts: sampling, shot exclusion, legs, offline report build."""

from __future__ import annotations

from pl_jobs_lora.config import load_config
from pl_jobs_lora.dataset.labeling_qa import (
    build_report,
    human_leg,
    select_sample,
    split_shots_eval,
    to_dev_examples,
)


def _gold(**over):
    base = {
        "title": None, "seniority": [], "work_mode": [], "tech_expected": [],
        "tech_optional": [], "contract_types": [], "salary": None,
    }
    return {**base, **over}


def _record(offer_id, **gold_over):
    return {
        "offer_id": offer_id, "url": f"https://x/{offer_id}", "pub_date": "2026-01-01T00:00:00Z",
        "prose": f"prose for {offer_id}", "gold": _gold(**gold_over),
    }


# -- sampling --------------------------------------------------------------------------------------

def test_select_sample_deterministic_and_sized():
    ids = [f"id{i}" for i in range(20)]
    a = select_sample(ids, {"id0"}, 8, seed=42, strategy="stratified")
    b = select_sample(ids, {"id0"}, 8, seed=42, strategy="stratified")
    assert a == b  # same seed → identical draw
    assert len(a) == 8
    assert set(a) <= set(ids)  # never invents ids outside the pool (excluded shots stay out)


def test_select_sample_different_seed_differs():
    ids = [f"id{i}" for i in range(50)]
    a = select_sample(ids, set(), 10, seed=1, strategy="random")
    b = select_sample(ids, set(), 10, seed=2, strategy="random")
    assert a != b


def test_select_sample_overweights_disagreements():
    ids = [f"id{i}" for i in range(10)]
    disagreements = {"id0", "id1"}  # base rate 0.2
    sample = select_sample(ids, disagreements, 4, seed=1, strategy="stratified")
    picked = set(sample) & disagreements
    # the disagreement stratum is over-represented vs its base rate in the full pool
    assert len(picked) / len(sample) > len(disagreements) / len(ids)
    assert disagreements <= set(sample)  # both scarce disagreements are pulled in


def test_split_shots_eval_excludes_shots():
    train = [_record("t0"), _record("t1"), _record("t2")]
    test = [_record("e0")]
    shots, eval_recs = split_shots_eval(train, test, n_shots=2)
    shot_ids = {r["offer_id"] for r in shots}
    assert shot_ids == {"t0", "t1"}
    assert shot_ids.isdisjoint({r["offer_id"] for r in eval_recs})
    assert {r["offer_id"] for r in eval_recs} == {"t2", "e0"}


# -- legs ------------------------------------------------------------------------------------------

def test_to_dev_examples_roundtrip():
    rec = _record("x", title="Dev", seniority=["mid"])
    ex = to_dev_examples([rec])[0]
    assert ex.offer_id == "x" and ex.prose == "prose for x"
    assert ex.gold["title"] == "Dev" and ex.gold["seniority"] == ["mid"]


def test_human_leg_accepts_nested_and_flat():
    nested = {"offer_id": "a", "human_gold": {"title": "T", "seniority": ["mid"]}}
    flat = {"offer_id": "b", "title": "U", "seniority": ["senior"]}
    legs = human_leg([nested, flat])
    assert legs[0] == {"offer_id": "a", "title": "T", "seniority": ["mid"]}
    assert legs[1] == {"offer_id": "b", "title": "U", "seniority": ["senior"]}


# -- offline report build (no model, no network) ---------------------------------------------------

def test_build_report_end_to_end(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    _write_jsonl(processed / "train.jsonl", [
        _record("a", seniority=["mid"], title="A"),
        _record("b", seniority=["senior"], title="B"),
    ])
    _write_jsonl(processed / "test.jsonl", [_record("c", seniority=["junior"], title="C")])

    proposals = tmp_path / "proposals.jsonl"
    _write_jsonl(proposals, [
        {"offer_id": "a", "valid": True, "parsed": {"title": "A", "seniority": ["mid"]}},
        {"offer_id": "b", "valid": True, "parsed": {"title": "B", "seniority": ["mid"]}},  # differs
        {"offer_id": "c", "valid": True, "parsed": {"title": "C", "seniority": ["junior"]}},
    ])
    human = tmp_path / "human_gold.jsonl"
    _write_jsonl(human, [
        {"offer_id": "c", "human_gold": {"title": "C", "seniority": ["junior"]}},
    ])

    cfg = load_config()
    report = build_report(
        cfg, processed_dir=processed, proposals_path=proposals, human_path=human,
    )
    d = report.as_dict()
    assert d["llm_vs_platform"]["n"] == 3       # matched all three proposed records
    assert d["metadata"]["n_sample"] == 1       # one human-checked record
    assert d["triangulation"]["n"] == 1
    # id c: llm==platform==human on seniority → all_agree (id b isn't in the human sample)
    assert d["triangulation"]["per_field"]["seniority"]["all_agree"] == 1


def test_build_report_without_human_has_only_full_leg(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    _write_jsonl(processed / "train.jsonl", [_record("a", seniority=["mid"])])
    _write_jsonl(processed / "test.jsonl", [])
    proposals = tmp_path / "proposals.jsonl"
    _write_jsonl(proposals, [{"offer_id": "a", "valid": True, "parsed": {"seniority": ["mid"]}}])

    cfg = load_config()
    report = build_report(
        cfg, processed_dir=processed, proposals_path=proposals,
        human_path=tmp_path / "absent.jsonl",
    )
    d = report.as_dict()
    assert d["llm_vs_platform"]["n"] == 1
    assert d["llm_vs_human"] is None and d["triangulation"] is None


def _write_jsonl(path, rows):
    import json
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

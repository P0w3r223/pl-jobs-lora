"""Freeze layer: JSONL round-trip + manifest shape (offline, tmp_path)."""

from __future__ import annotations

import json

from pl_jobs_lora.config import load_config
from pl_jobs_lora.dataset.hf_dataset import dataset_manifest, write_jsonl


def _rec(offer_id: str, pub_date: str) -> dict:
    return {"offer_id": offer_id, "url": f"https://x/{offer_id}", "pub_date": pub_date,
            "prose": "Zakres obowiązków: Python.", "gold": {"title": "Dev"}}


def test_write_jsonl_roundtrip_preserves_polish(tmp_path):
    records = [_rec("a", "2026-07-01T00:00:00Z"), _rec("b", "2026-07-02T00:00:00Z")]
    path = tmp_path / "sub" / "train.jsonl"  # parent created by write_jsonl
    write_jsonl(records, path)
    back = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert back == records
    assert "obowiązków" in path.read_text(encoding="utf-8")  # ensure_ascii=False kept


def test_manifest_reports_counts_and_date_ranges():
    cfg = load_config()
    train = [_rec("a", "2026-06-01T00:00:00Z"), _rec("b", "2026-06-10T00:00:00Z")]
    test = [_rec("c", "2026-07-01T00:00:00Z")]
    stats = {"collected": 5, "passed_filters": 4, "dropped_filters": 1,
             "dropped_duplicates": 1, "records": 3}
    m = dataset_manifest(train, test, stats, cfg)
    assert m["splits"]["train"] == {"n": 2, "dates": {
        "earliest": "2026-06-01T00:00:00Z", "latest": "2026-06-10T00:00:00Z"}}
    assert m["splits"]["test"] == {"n": 1, "dates": {
        "earliest": "2026-07-01T00:00:00Z", "latest": "2026-07-01T00:00:00Z"}}
    assert m["build"] == stats
    assert m["params"]["test_fraction"] == cfg.data.test_fraction


def test_manifest_handles_empty_test():
    cfg = load_config()
    m = dataset_manifest([_rec("a", "2026-06-01T00:00:00Z")], [], {"records": 1}, cfg)
    assert m["splits"]["test"] == {"n": 0, "dates": None}

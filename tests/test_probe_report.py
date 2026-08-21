"""Probe aggregation (pure, no llama-cpp): variant report, winner pick, table render, rescore."""

from __future__ import annotations

import json

from pl_jobs_lora.config import load_config
from pl_jobs_lora.probe import (
    build_variant_report,
    pick_winner,
    read_predictions,
    render_table,
    write_probe_report,
)

_CFG = load_config()
_GOLD = [{"offer_id": "a", "tech_expected": ["python"], "seniority": ["mid"],
          "work_mode": ["remote"], "tech_optional": [], "salary": None, "title": "X"}]


def _preds(parsed, valid, latency, tokens):
    return [{"offer_id": "a", "valid": valid, "parsed": parsed,
             "latency_s": latency, "output_tokens": tokens}]


def test_variant_report_folds_scores_and_economy():
    r = build_variant_report(_preds(dict(_GOLD[0]), True, 2.0, 80), _GOLD, _CFG)
    assert r["mean_field_f1"] == 1.0
    assert r["json_validity"] == 1.0
    assert r["latency_p50_s"] == 2.0
    assert r["mean_output_tokens"] == 80.0


def test_pick_winner_prefers_accuracy_then_latency():
    good = build_variant_report(_preds(dict(_GOLD[0]), True, 5.0, 80), _GOLD, _CFG)
    bad = build_variant_report(_preds(None, False, 1.0, 0), _GOLD, _CFG)
    reports = {"cand-a/few": good, "cand-b/zero": bad}
    assert pick_winner(reports) == "cand-a/few"


def test_render_table_has_header_and_rows():
    r = build_variant_report(_preds(dict(_GOLD[0]), True, 2.0, 80), _GOLD, _CFG)
    table = render_table({"cand-a/few": r})
    assert "variant" in table and "cand-a/few" in table


def test_render_table_dashes_unmeasured_latency_and_tokens():
    """A variant with no timing must render `-`, not the literal `None`."""
    preds = [{"offer_id": "a", "valid": True, "parsed": dict(_GOLD[0])}]  # no latency/tokens
    table = render_table({"cand-a/few": build_variant_report(preds, _GOLD, _CFG)})
    assert "None" not in table


def test_read_predictions_returns_none_for_a_variant_never_run(tmp_path):
    assert read_predictions("cand-a", "few", pred_dir=tmp_path) is None


def test_read_predictions_round_trips_a_stored_file(tmp_path):
    rows = [{"offer_id": "a", "valid": True, "parsed": {"seniority": ["mid"]}}]
    (tmp_path / "cand-a__few.jsonl").write_text(
        json.dumps(rows[0], ensure_ascii=False) + "\n", encoding="utf-8"
    )
    assert read_predictions("cand-a", "few", pred_dir=tmp_path) == rows


def test_write_probe_report_records_the_winner_in_both_artifacts(tmp_path):
    """The rescore path must produce the same two artifacts a live probe run does."""
    good = build_variant_report(_preds(dict(_GOLD[0]), True, 5.0, 80), _GOLD, _CFG)
    bad = build_variant_report(_preds(None, False, 1.0, 0), _GOLD, _CFG)
    out = write_probe_report({"cand-a/few": good, "cand-b/zero": bad}, n_eval=1, out_dir=tmp_path)

    assert out["winner"] == "cand-a/few" and out["n_eval"] == 1
    written = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert written == out
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "cand-a/few" in md and "n=1" in md

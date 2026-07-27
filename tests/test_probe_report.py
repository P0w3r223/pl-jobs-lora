"""Probe aggregation (pure, no llama-cpp): variant report, winner pick, table render."""

from __future__ import annotations

from pl_jobs_lora.config import load_config
from pl_jobs_lora.probe import build_variant_report, pick_winner, render_table

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

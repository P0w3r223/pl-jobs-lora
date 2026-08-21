"""Probe aggregation (pure, no llama-cpp): variant report, winner pick, table render, rescore."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from pl_jobs_lora.config import load_config
from pl_jobs_lora.probe import (
    build_variant_report,
    check_context_fits,
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


class _FakeLlm:
    """Stands in for llama-cpp: one token per whitespace-separated word is enough here."""

    @staticmethod
    def tokenize(blob: bytes) -> list[int]:
        return [0] * len(blob.decode("utf-8").split())


def _prompt(n_tokens):
    return [{"role": "user", "content": " ".join("x" * 1 for _ in range(n_tokens))}]


def _cfg_with(**probe):
    return replace(_CFG, probe=replace(_CFG.probe, **probe))


def test_context_fit_passes_when_the_longest_prompt_leaves_room():
    cfg = _cfg_with(context_tokens=1000, max_tokens=400, context_margin_tokens=100)
    worst = check_context_fits(_FakeLlm, [_prompt(120), _prompt(500)], cfg)
    assert worst == 500, "the check reports the prompt that came closest to the limit"


def test_context_fit_refuses_a_cap_the_longest_prompt_cannot_afford():
    """Caught before the first token is generated, not at record 90 of a four-hour run."""
    cfg = _cfg_with(context_tokens=1000, max_tokens=600, context_margin_tokens=100)
    with pytest.raises(ValueError, match="exceeds the 300 left by n_ctx=1000"):
        check_context_fits(_FakeLlm, [_prompt(120), _prompt(301)], cfg)


def test_context_fit_counts_the_margin_against_the_budget():
    """The tokenizer sees message contents; the chat template adds role markers on top."""
    cfg = _cfg_with(context_tokens=1000, max_tokens=600, context_margin_tokens=0)
    assert check_context_fits(_FakeLlm, [_prompt(400)], cfg) == 400
    with pytest.raises(ValueError):
        check_context_fits(_FakeLlm, [_prompt(400)], _cfg_with(
            context_tokens=1000, max_tokens=600, context_margin_tokens=1))


def test_context_fit_is_vacuous_for_an_empty_eval_set():
    assert check_context_fits(_FakeLlm, [], _CFG) == 0

"""Comparison report (ADR-0003): scoring rollup, cost/latency folding, ordering, rendering.

Pure and offline — synthetic predictions on disk, no model, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

from pl_jobs_lora.eval import scoring
from pl_jobs_lora.eval.report import build_report, score_variant, tally_failures

_PRICING = {"input_usd_per_mtok": 1.0, "output_usd_per_mtok": 5.0}
_PCTL = (50, 95)


def _gold(oid, **over):
    base = {
        "offer_id": oid, "title": None, "seniority": [], "work_mode": [],
        "tech_expected": [], "tech_optional": [], "salary": None,
    }
    return {**base, **over}


def _pred(oid, parsed, *, valid=True, in_tok=None, out_tok=None, latency=None):
    row = {"offer_id": oid, "valid": valid, "parsed": parsed}
    if in_tok is not None:
        row["input_tokens"] = in_tok
        row["output_tokens"] = out_tok
    if latency is not None:
        row["latency_s"] = latency
    return row


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


_GOLD = [
    _gold("a", seniority=["mid"], work_mode=["remote"], tech_expected=["python"], title="A"),
    _gold("b", seniority=["senior"], work_mode=["office"], tech_expected=["java"], title="B"),
]


def _score(preds):
    return score_variant(
        "v", preds, _GOLD, salary_rel_tolerance=0.05,
        latency_percentiles=_PCTL, **_PRICING,
    )


def test_perfect_variant_scores_top():
    perfect = [
        _pred("a", {"seniority": ["mid"], "work_mode": ["remote"],
                    "tech_expected": ["python"], "title": "A"}),
        _pred("b", {"seniority": ["senior"], "work_mode": ["office"],
                    "tech_expected": ["java"], "title": "B"}),
    ]
    r = _score(perfect)
    assert r.scores["json_validity"] == 1.0
    assert r.scores["fields"]["seniority"]["f1"] == 1.0
    assert r.mean_field_f1 == 1.0


def test_worse_predictions_score_lower():
    """The regression property (ADR-0003): degraded predictions must show visibly lower F1."""
    good = [_pred("a", {"seniority": ["mid"]}), _pred("b", {"seniority": ["senior"]})]
    bad = [_pred("a", {"seniority": ["junior"]}), _pred("b", {"seniority": ["junior"]})]
    assert _score(good).scores["fields"]["seniority"]["f1"] > \
        _score(bad).scores["fields"]["seniority"]["f1"]


def test_invalid_json_lowers_validity():
    r = _score([_pred("a", None, valid=False), _pred("b", {"seniority": ["senior"]}, valid=True)])
    assert r.scores["json_validity"] == 0.5


def test_api_variant_prices_out_local_variant_free():
    api = [
        _pred("a", {"seniority": ["mid"]}, in_tok=1000, out_tok=200, latency=0.4),
        _pred("b", {"seniority": ["senior"]}, in_tok=1000, out_tok=200, latency=0.8),
    ]
    r = _score(api)
    # per call = (1000*1 + 200*5)/1e6 = 0.002 USD -> $2.00 per 1000 postings
    assert r.usd_per_1k_postings == 2.0
    assert r.latency_s == {"p50": 0.6, "p95": 0.78}

    local = [_pred("a", {"seniority": ["mid"]}), _pred("b", {"seniority": ["senior"]})]
    assert _score(local).usd_per_1k_postings is None  # no token counts -> ~$0 local run


def test_build_report_aggregates_all_variant_files(tmp_path: Path):
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    _write_jsonl(pred_dir / "claude-haiku-4-5__zero.jsonl", [
        _pred("a", {"seniority": ["mid"]}, in_tok=900, out_tok=100, latency=0.5),
        _pred("b", {"seniority": ["senior"]}, in_tok=900, out_tok=100, latency=0.5),
    ])
    _write_jsonl(pred_dir / "base.jsonl", [
        _pred("a", None, valid=False), _pred("b", {"seniority": ["junior"]}),
    ])

    report = build_report(
        sorted(pred_dir.glob("*.jsonl")), _GOLD,
        salary_rel_tolerance=0.05, latency_percentiles=_PCTL, api_pricing="haiku", **_PRICING,
    )
    d = report.as_dict()
    assert d["metadata"]["n_gold"] == 2
    assert d["metadata"]["n_variants"] == 2
    names = [v["variant"] for v in d["variants"]]
    assert names == ["base", "claude-haiku-4-5__zero"]  # sorted by filename
    md = report.render_markdown()
    assert "claude-haiku-4-5__zero" in md and "| base |" in md
    assert md.count("\n") >= 6  # header block + 2 data rows


def test_tally_failures_ignores_valid_rows_and_orders_by_taxonomy():
    preds = [
        _pred("a", {"seniority": ["mid"]}),                                    # valid
        {"offer_id": "b", "valid": False, "parsed": None,
         "failure": scoring.SCHEMA_INVALID},
        {"offer_id": "c", "valid": False, "parsed": None,
         "failure": scoring.NO_JSON_OBJECT},
        {"offer_id": "d", "valid": False, "parsed": None,
         "failure": scoring.NO_JSON_OBJECT},
    ]
    counts, at_cap = tally_failures(preds, decode_max_tokens=1024)
    assert counts == {scoring.NO_JSON_OBJECT: 2, scoring.SCHEMA_INVALID: 1}
    assert list(counts) == [scoring.NO_JSON_OBJECT, scoring.SCHEMA_INVALID]  # taxonomy order
    assert at_cap == 0


def test_a_row_predating_the_taxonomy_counts_as_unrecorded():
    """An old predictions file must not read as one that parsed cleanly."""
    preds = [{"offer_id": "a", "valid": False, "parsed": None}]   # no `failure` key
    counts, _ = tally_failures(preds, decode_max_tokens=1024)
    assert counts == {scoring.UNRECORDED: 1}


def test_a_class_this_build_no_longer_knows_is_unrecorded_not_dropped():
    """A vocabulary that drifted must degrade to "reason unknown", never to a missing row.

    `not_an_object` was a real class earlier on this branch, so files written then carry it.
    Projecting the histogram through the current taxonomy silently discarded such rows — which is
    the exact guarantee `unrecorded` exists to provide, failing when the vocabulary changed rather
    than when it was absent.
    """
    preds = [
        {"offer_id": "a", "valid": False, "parsed": None, "failure": "not_an_object"},
        {"offer_id": "b", "valid": False, "parsed": None, "failure": scoring.EMPTY_OUTPUT},
    ]
    counts, _ = tally_failures(preds, decode_max_tokens=1024)
    assert counts == {scoring.EMPTY_OUTPUT: 1, scoring.UNRECORDED: 1}
    assert sum(counts.values()) == 2, "every invalid row must survive into the histogram"


def test_predictions_outside_the_gold_set_are_not_tallied():
    """The failure table must not list failures the main table says do not exist."""
    preds = [
        {"offer_id": "a", "valid": False, "parsed": None, "failure": scoring.NO_JSON_OBJECT},
        {"offer_id": "ghost", "valid": False, "parsed": None,
         "failure": scoring.NO_JSON_OBJECT},          # not in gold: invisible to the scorer
    ]
    counts, _ = tally_failures(preds, decode_max_tokens=1024, gold_ids={"a", "b"})
    assert counts == {scoring.NO_JSON_OBJECT: 1}


def test_the_invalid_column_agrees_with_the_scorer(tmp_path: Path):
    """Two code paths build the two tables; they must not disagree on how many rows failed."""
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    _write_jsonl(pred_dir / "v.jsonl", [
        _pred("a", {"seniority": ["mid"]}),
        {"offer_id": "b", "valid": False, "parsed": None, "failure": scoring.NO_JSON_OBJECT},
        {"offer_id": "ghost", "valid": False, "parsed": None,
         "failure": scoring.NO_JSON_OBJECT},
    ])
    v = build_report(
        sorted(pred_dir.glob("*.jsonl")), _GOLD, salary_rel_tolerance=0.05,
        latency_percentiles=_PCTL, api_pricing="haiku", decode_max_tokens=1024, **_PRICING,
    ).variants[0]
    scored_invalid = round(v.scores["n"] * (1 - v.scores["json_validity"]))
    assert sum(v.failures.values()) == scored_invalid == 1


def test_decode_failures_at_the_token_cap_are_separated_from_malformed_ones():
    """The truncation confound: same failure class, different cause, and validity cannot tell."""
    preds = [
        {"offer_id": "a", "valid": False, "parsed": None,
         "failure": scoring.JSON_DECODE_ERROR, "output_tokens": 1024},   # hit the cap
        {"offer_id": "b", "valid": False, "parsed": None,
         "failure": scoring.JSON_DECODE_ERROR, "output_tokens": 40},     # genuinely malformed
        {"offer_id": "c", "valid": False, "parsed": None,
         "failure": scoring.NO_JSON_OBJECT, "output_tokens": 1024},      # capped, but not a decode
    ]
    counts, at_cap = tally_failures(preds, decode_max_tokens=1024)
    assert counts[scoring.JSON_DECODE_ERROR] == 2
    assert at_cap == 1


def test_a_rows_own_cap_beats_the_callers_default():
    """GGUF decodes under probe.max_tokens and the API under eval.max_tokens.

    One global value would attribute a truncation against a limit the variant never ran with —
    and this cross-tab is now used to decide whether a failure is the model's or the harness's.
    """
    preds = [
        {"offer_id": "a", "valid": False, "parsed": None, "failure": scoring.JSON_DECODE_ERROR,
         "output_tokens": 512, "max_tokens": 512},      # capped under its own, smaller limit
        {"offer_id": "b", "valid": False, "parsed": None, "failure": scoring.JSON_DECODE_ERROR,
         "output_tokens": 512, "max_tokens": 2048},     # nowhere near its own limit
    ]
    assert tally_failures(preds, decode_max_tokens=1024)[1] == 1


def test_a_row_without_its_own_cap_falls_back_to_the_default():
    """Rows written before the cap was recorded still cross-tab against the configured value."""
    preds = [{"offer_id": "a", "valid": False, "parsed": None,
              "failure": scoring.JSON_DECODE_ERROR, "output_tokens": 1024}]
    assert tally_failures(preds, decode_max_tokens=1024)[1] == 1


def test_token_cap_cross_tab_is_skipped_when_the_cap_is_unknown():
    preds = [{"offer_id": "a", "valid": False, "parsed": None,
              "failure": scoring.JSON_DECODE_ERROR, "output_tokens": 1024}]
    assert tally_failures(preds, decode_max_tokens=None)[1] == 0


def test_report_renders_the_failure_taxonomy_section(tmp_path: Path):
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    _write_jsonl(pred_dir / "v.jsonl", [
        _pred("a", {"seniority": ["mid"]}),
        {"offer_id": "b", "valid": False, "parsed": None,
         "failure": scoring.NO_JSON_OBJECT},
    ])
    report = build_report(
        sorted(pred_dir.glob("*.jsonl")), _GOLD, salary_rel_tolerance=0.05,
        latency_percentiles=_PCTL, api_pricing="haiku", decode_max_tokens=1024, **_PRICING,
    )
    assert report.variants[0].failures == {scoring.NO_JSON_OBJECT: 1}
    md = report.render_markdown()
    assert "## Failure taxonomy" in md and scoring.NO_JSON_OBJECT in md


def test_failure_section_says_so_when_nothing_failed(tmp_path: Path):
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    _write_jsonl(pred_dir / "v.jsonl", [_pred("a", {"seniority": ["mid"]})])
    report = build_report(
        sorted(pred_dir.glob("*.jsonl")), _GOLD, salary_rel_tolerance=0.05,
        latency_percentiles=_PCTL, api_pricing="haiku", **_PRICING,
    )
    assert "parsed cleanly on every record" in report.render_markdown()


def test_a_complete_run_carries_no_warning_marker():
    complete = [_pred("a", {"seniority": ["mid"]}), _pred("b", {"seniority": ["senior"]})]
    v = _score(complete)
    assert v.trustworthy and v.coverage == 1.0 and v.n_duplicate_rows == 0


def test_a_duplicated_row_is_flagged_even_at_full_coverage(tmp_path: Path):
    """Full coverage must not launder a predictions file that answers one record twice."""
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    _write_jsonl(pred_dir / "dupes.jsonl", [
        _pred("a", {"seniority": ["mid"]}),
        _pred("b", {"seniority": ["senior"]}),
        _pred("b", {"seniority": ["senior"]}),   # appended twice
    ])
    report = build_report(
        sorted(pred_dir.glob("*.jsonl")), _GOLD,
        salary_rel_tolerance=0.05, latency_percentiles=_PCTL, api_pricing="haiku", **_PRICING,
    )
    v = report.variants[0]
    assert v.coverage == 1.0, "the file does answer every record — coverage alone looks clean"
    assert v.n_duplicate_rows == 1 and not v.trustworthy
    assert report.as_dict()["metadata"]["duplicated_row_variants"] == {"dupes": 1}
    assert "⚠" in report.render_markdown()

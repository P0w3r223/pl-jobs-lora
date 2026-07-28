"""Comparison report (ADR-0003): scoring rollup, cost/latency folding, ordering, rendering.

Pure and offline — synthetic predictions on disk, no model, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

from pl_jobs_lora.eval.report import build_report, score_variant

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

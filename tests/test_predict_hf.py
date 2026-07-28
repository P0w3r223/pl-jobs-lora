"""HF inference assembly (S5; ADR-0006): prediction rows, shot modes, no local token cost.

Offline — ``generate_fn`` is injected, so transformers/peft and the GPU are never touched.
"""

from __future__ import annotations

import json

from pl_jobs_lora.config import load_config
from pl_jobs_lora.dataset.collect import DevExample
from pl_jobs_lora.inference.predict_hf import run_inference, run_predictions


def _gold(**over):
    base = {
        "title": None, "seniority": [], "work_mode": [], "tech_expected": [],
        "tech_optional": [], "contract_types": [], "salary": None,
    }
    return {**base, **over}


def _example(oid):
    return DevExample(offer_id=oid, url=f"https://x/{oid}", pub_date=None,
                      prose=f"prose {oid}", gold=_gold())


def _record(oid):
    return {"offer_id": oid, "url": f"https://x/{oid}", "pub_date": None,
            "prose": f"prose {oid}", "gold": _gold()}


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_run_inference_assembles_rows_without_token_counts():
    cfg = load_config()
    preds = run_inference(
        cfg, [_example("a"), _example("b")], [], mode="zero",
        generate_fn=lambda m: json.dumps({"title": "Dev", "seniority": ["mid"]}),
    )
    assert [p["offer_id"] for p in preds] == ["a", "b"]
    assert preds[0]["valid"] is True and preds[0]["parsed"]["title"] == "Dev"
    assert "latency_s" in preds[0]
    assert "input_tokens" not in preds[0]  # a local run is ~$0 — the report treats it as no cost


def test_run_inference_invalid_output_marks_invalid():
    cfg = load_config()
    preds = run_inference(
        cfg, [_example("a")], [], mode="zero", generate_fn=lambda m: "rambling, no json",
    )
    assert preds[0]["valid"] is False and preds[0]["parsed"] is None


def test_run_inference_few_mode_adds_shots():
    cfg = load_config()  # eval.few_shot_examples == 2
    shots = [_example("s0"), _example("s1")]
    seen: dict[str, list] = {}

    def recorder(mode):
        def fake(messages):
            seen[mode] = messages
            return "{}"
        return fake

    run_inference(cfg, [_example("a")], shots, mode="zero", generate_fn=recorder("zero"))
    run_inference(cfg, [_example("a")], shots, mode="few", generate_fn=recorder("few"))
    assert len(seen["few"]) > len(seen["zero"])  # shots injected only in few mode


def test_run_predictions_variants_and_base_before_adapter(tmp_path):
    """Fairness wiring (offline via seams): variant naming + base runs before the adapter is on."""
    processed = tmp_path / "processed"
    processed.mkdir()
    _write_jsonl(processed / "train.jsonl", [_record("t0"), _record("t1"), _record("t2")])
    _write_jsonl(processed / "test.jsonl", [_record("e0"), _record("e1")])
    pred_dir = tmp_path / "predictions"

    seen: list[str] = []

    def gen_factory(model):
        def gen(messages):
            seen.append(model)
            return json.dumps({"title": "X"})
        return gen

    out = run_predictions(
        load_config(), base=True, lora=True, processed_dir=processed, pred_dir=pred_dir,
        load_fn=lambda cfg: ("BASE", "TOK"), attach_fn=lambda model, repo: "LORA",
        generate_factory=gen_factory,
    )
    assert set(out) == {"bielik-1.5b__zero", "bielik-1.5b__few", "bielik-1.5b-lora__zero"}
    assert (pred_dir / "bielik-1.5b-lora__zero.jsonl").exists()
    # base (adapter off) is generated before the LoRA variant (adapter on), from one load
    assert "LORA" in seen and seen.index("BASE") < seen.index("LORA")

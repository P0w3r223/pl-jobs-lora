"""API baseline runner assembly (ADR-0003): shared prompt, egress cap, per-variant files.

Offline — ``generate_fn`` is injected, so no Anthropic client and no network are touched.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from pl_jobs_lora.config import load_config
from pl_jobs_lora.dataset.collect import DevExample
from pl_jobs_lora.eval.baselines import _api_generate, run_baseline_inference, run_baselines


class _Block:
    def __init__(self, type, text=None):
        self.type = type
        self.text = text


class _Usage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _Resp:
    def __init__(self, content, usage):
        self.content = content
        self.usage = usage


class _FakeClient:
    """Duck-typed stand-in for anthropic.Anthropic — records the call, returns a canned response."""

    def __init__(self, resp):
        self._resp = resp
        self.messages = self

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._resp


def _gold(**over):
    base = {
        "title": None, "seniority": [], "work_mode": [], "tech_expected": [],
        "tech_optional": [], "contract_types": [], "salary": None,
    }
    return {**base, **over}


def _example(oid):
    return DevExample(offer_id=oid, url=f"https://x/{oid}", pub_date=None, prose=f"prose {oid}",
                      gold=_gold())


def _record(oid):
    return {"offer_id": oid, "url": f"https://x/{oid}", "pub_date": None,
            "prose": f"prose {oid}", "gold": _gold()}


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_run_baseline_inference_assembles_predictions():
    cfg = load_config()
    eval_set = [_example("a"), _example("b")]

    def fake(messages):
        return json.dumps({"title": "Dev", "seniority": ["mid"]}), 1200, 300

    preds = run_baseline_inference(cfg, "zero", eval_set, [], generate_fn=fake)
    assert [p["offer_id"] for p in preds] == ["a", "b"]
    assert preds[0]["valid"] is True
    assert preds[0]["parsed"]["title"] == "Dev" and preds[0]["parsed"]["seniority"] == ["mid"]
    assert preds[0]["input_tokens"] == 1200 and preds[0]["output_tokens"] == 300
    assert "latency_s" in preds[0]


def test_run_baseline_inference_invalid_json_marks_invalid():
    cfg = load_config()
    preds = run_baseline_inference(
        cfg, "zero", [_example("a")], [], generate_fn=lambda m: ("not json at all", 5, 5),
    )
    assert preds[0]["valid"] is False and preds[0]["parsed"] is None


def test_few_mode_adds_shots_zero_mode_does_not():
    cfg = load_config()  # eval.few_shot_examples == 2
    eval_set = [_example("a")]
    shots = [_example("s0"), _example("s1")]
    seen: dict[str, list] = {}

    def recorder(mode):
        def fake(messages):
            seen[mode] = messages
            return "{}", 1, 1
        return fake

    run_baseline_inference(cfg, "zero", eval_set, shots, generate_fn=recorder("zero"))
    run_baseline_inference(cfg, "few", eval_set, shots, generate_fn=recorder("few"))
    assert len(seen["few"]) > len(seen["zero"])  # shots injected only in few mode


def test_run_baseline_inference_respects_egress_cap():
    cfg = load_config()
    ev = dataclasses.replace(cfg.eval, request_max_snippets=1)
    cfg = dataclasses.replace(cfg, eval=ev)
    with pytest.raises(ValueError, match="egress cap"):
        run_baseline_inference(
            cfg, "zero", [_example("a"), _example("b")], [], generate_fn=lambda m: ("{}", 1, 1),
        )


def test_run_baselines_writes_a_file_per_variant(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    _write_jsonl(processed / "train.jsonl", [_record("t0"), _record("t1"), _record("t2")])
    _write_jsonl(processed / "test.jsonl", [_record("e0"), _record("e1")])
    pred_dir = tmp_path / "predictions"

    cfg = load_config()
    out = run_baselines(
        cfg, processed_dir=processed, pred_dir=pred_dir,
        generate_fn=lambda m: (json.dumps({"title": "X"}), 10, 5),
    )
    assert set(out) == {"claude-haiku-4-5__zero", "claude-haiku-4-5__few"}
    assert (pred_dir / "claude-haiku-4-5__zero.jsonl").exists()
    # eval set is the frozen test split (2 records); shots are the train head, excluded from eval
    assert len(out["claude-haiku-4-5__zero"]) == 2
    assert [p["offer_id"] for p in out["claude-haiku-4-5__few"]] == ["e0", "e1"]


# -- response mapping: the lines that only run against a real Anthropic response ------------------

def test_api_generate_joins_text_and_reads_usage():
    cfg = load_config()
    resp = _Resp([_Block("text", '{"title": "Dev"}')], _Usage(1200, 300))
    client = _FakeClient(resp)
    messages = [{"content": "sys"}, {"role": "user", "content": "x"}]
    raw, in_tok, out_tok = _api_generate(client, cfg, messages)
    assert raw == '{"title": "Dev"}' and in_tok == 1200 and out_tok == 300
    # the system block is split off; the model + decoding caps come from config (fairness)
    assert client.last_kwargs["model"] == cfg.eval.api_model
    assert client.last_kwargs["system"] == "sys"
    assert client.last_kwargs["temperature"] == cfg.eval.temperature


def test_api_generate_concatenates_text_and_skips_non_text_blocks():
    cfg = load_config()
    resp = _Resp([_Block("text", "a"), _Block("thinking"), _Block("text", "b")], _Usage(1, 1))
    messages = [{"content": "s"}, {"role": "user", "content": "x"}]
    raw, _, _ = _api_generate(_FakeClient(resp), cfg, messages)
    assert raw == "ab"

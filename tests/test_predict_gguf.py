"""Resumable GGUF backfill: stop a multi-hour CPU run and finish it later without redoing work.

The model is monkeypatched out — this exercises the resume bookkeeping, not llama-cpp.
"""

from __future__ import annotations

import contextlib
import json

from pl_jobs_lora.config import load_config
from pl_jobs_lora.inference import predict_gguf as pg


def _record(offer_id, **gold):
    base = {
        "title": None, "seniority": [], "work_mode": [],
        "tech_expected": [], "tech_optional": [], "salary": None,
    }
    return {
        "offer_id": offer_id, "url": f"https://x/{offer_id}", "pub_date": "2026-01-01T00:00:00",
        "prose": f"prose for {offer_id}", "gold": {**base, **gold},
    }


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _seed(monkeypatch, tmp_path, *, fail_after=None):
    """Point the run at a tmp frozen set; record which ids inference is actually asked for."""
    cfg = load_config()
    n_shots = cfg.probe.few_shot_examples
    processed = tmp_path / "processed"
    _write_jsonl(processed / "train.jsonl", [_record(f"t{i}") for i in range(n_shots + 1)])
    _write_jsonl(processed / "test.jsonl", [_record(f"e{i}") for i in range(4)])
    pred_dir = tmp_path / "predictions"

    asked: list[list[str]] = []

    def fake_run_inference(cand, mode, eval_set, shots, cfg_, on_prediction=None):
        asked.append([ex.offer_id for ex in eval_set])
        preds = []
        for i, ex in enumerate(eval_set):
            if fail_after is not None and i == fail_after:
                raise KeyboardInterrupt
            pred = {
                "offer_id": ex.offer_id, "valid": True, "parsed": {"title": ex.offer_id},
                "failure": None, "raw": '{"title": "x"}', "latency_s": 0.1, "output_tokens": 3,
            }
            if on_prediction is not None:
                on_prediction(pred)
            preds.append(pred)
        return preds

    monkeypatch.setattr("pl_jobs_lora.probe.run_inference", fake_run_inference)
    monkeypatch.setattr(pg, "resolve_base", lambda c: type("B", (), {"key": "base"})())
    return cfg, processed, pred_dir, asked


def test_a_first_run_computes_every_record(monkeypatch, tmp_path):
    cfg, processed, pred_dir, asked = _seed(monkeypatch, tmp_path)
    variant, preds = pg.run_gguf_predictions(
        cfg, mode="few", processed_dir=processed, pred_dir=pred_dir,
    )
    assert variant == "base-gguf__few"
    assert asked == [["e0", "e1", "e2", "e3"]]
    assert [p["offer_id"] for p in preds] == ["e0", "e1", "e2", "e3"]


def test_a_completed_run_reruns_nothing(monkeypatch, tmp_path):
    cfg, processed, pred_dir, asked = _seed(monkeypatch, tmp_path)
    pg.run_gguf_predictions(cfg, mode="few", processed_dir=processed, pred_dir=pred_dir)
    asked.clear()
    _, preds = pg.run_gguf_predictions(
        cfg, mode="few", processed_dir=processed, pred_dir=pred_dir,
    )
    assert asked == [], "a finished run must not pay for itself twice"
    assert len(preds) == 4


def test_an_interrupted_run_resumes_where_it_stopped(monkeypatch, tmp_path):
    """The property the multi-hour backfill depends on."""
    cfg, processed, pred_dir, asked = _seed(monkeypatch, tmp_path, fail_after=2)
    with contextlib.suppress(KeyboardInterrupt):
        pg.run_gguf_predictions(cfg, mode="few", processed_dir=processed, pred_dir=pred_dir)
    path = pred_dir / "base-gguf__few.jsonl"
    assert [r["offer_id"] for r in _rows(path)] == ["e0", "e1"], "two records survived the kill"

    _, _, _, asked = _seed(monkeypatch, tmp_path)   # re-arm inference without the failure
    _, preds = pg.run_gguf_predictions(
        cfg, mode="few", processed_dir=processed, pred_dir=pred_dir,
    )
    assert asked == [["e2", "e3"]], "only the outstanding records are recomputed"
    assert [p["offer_id"] for p in preds] == ["e0", "e1", "e2", "e3"]


def test_rows_predating_the_taxonomy_are_rerun_not_resumed(monkeypatch, tmp_path):
    """How the failure taxonomy gets backfilled onto variants measured before it existed."""
    cfg, processed, pred_dir, asked = _seed(monkeypatch, tmp_path)
    path = pred_dir / "base-gguf__few.jsonl"
    _write_jsonl(path, [
        {"offer_id": f"e{i}", "valid": True, "parsed": {}, "latency_s": 9.9} for i in range(4)
    ])

    _, preds = pg.run_gguf_predictions(
        cfg, mode="few", processed_dir=processed, pred_dir=pred_dir,
    )
    assert asked == [["e0", "e1", "e2", "e3"]], "no pre-taxonomy row counts as done"
    assert all("failure" in p and "raw" in p for p in preds)


def test_superseded_rows_are_backed_up_before_being_discarded(monkeypatch, tmp_path):
    """They are regenerable — but regenerating them costs the hours resume exists to protect."""
    cfg, processed, pred_dir, _ = _seed(monkeypatch, tmp_path)
    path = pred_dir / "base-gguf__few.jsonl"
    legacy = [{"offer_id": f"e{i}", "valid": True, "parsed": {}, "latency_s": 9.9}
              for i in range(4)]
    _write_jsonl(path, legacy)

    pg.run_gguf_predictions(cfg, mode="few", processed_dir=processed, pred_dir=pred_dir)
    backup = path.with_suffix(path.suffix + ".pre-taxonomy")
    assert backup.exists()
    assert _rows(backup) == legacy, "the old measurement is recoverable"


def test_fresh_recomputes_even_a_complete_current_file(monkeypatch, tmp_path):
    cfg, processed, pred_dir, asked = _seed(monkeypatch, tmp_path)
    pg.run_gguf_predictions(cfg, mode="few", processed_dir=processed, pred_dir=pred_dir)
    asked.clear()
    pg.run_gguf_predictions(
        cfg, mode="few", processed_dir=processed, pred_dir=pred_dir, fresh=True,
    )
    assert asked == [["e0", "e1", "e2", "e3"]]


def test_a_completed_file_is_written_in_eval_order(monkeypatch, tmp_path):
    """Append order reflects when a run was interrupted; the artefact should not."""
    cfg, processed, pred_dir, _ = _seed(monkeypatch, tmp_path, fail_after=2)
    with contextlib.suppress(KeyboardInterrupt):
        pg.run_gguf_predictions(cfg, mode="few", processed_dir=processed, pred_dir=pred_dir)
    path = pred_dir / "base-gguf__few.jsonl"
    _write_jsonl(path, list(reversed(_rows(path))))       # simulate a scrambled resume order

    _seed(monkeypatch, tmp_path)
    pg.run_gguf_predictions(cfg, mode="few", processed_dir=processed, pred_dir=pred_dir)
    assert [r["offer_id"] for r in _rows(path)] == ["e0", "e1", "e2", "e3"]


def test_progress_is_reported_against_the_whole_set_not_just_the_todo(monkeypatch, tmp_path):
    """Resuming at record 130 of 142 should read as 131/142, not 1/12."""
    cfg, processed, pred_dir, _ = _seed(monkeypatch, tmp_path, fail_after=2)
    with contextlib.suppress(KeyboardInterrupt):
        pg.run_gguf_predictions(cfg, mode="few", processed_dir=processed, pred_dir=pred_dir)

    _seed(monkeypatch, tmp_path)
    seen: list[tuple[int, int]] = []
    pg.run_gguf_predictions(
        cfg, mode="few", processed_dir=processed, pred_dir=pred_dir,
        on_progress=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(3, 4), (4, 4)]

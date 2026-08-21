"""Resumable JSONL runs: what a killed multi-hour CPU job is allowed to lose, and what it isn't.

Pure and offline — no model, no network.
"""

from __future__ import annotations

import contextlib
import json

from pl_jobs_lora import resume


def _rows(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_load_completed_is_empty_for_a_run_that_never_started(tmp_path):
    assert resume.load_completed(tmp_path / "nothing.jsonl") == {}


def test_load_completed_keys_by_offer_id(tmp_path):
    path = tmp_path / "p.jsonl"
    resume.write_jsonl(path, [{"offer_id": "a", "v": 1}, {"offer_id": "b", "v": 2}])
    done = resume.load_completed(path)
    assert set(done) == {"a", "b"} and done["b"]["v"] == 2


def test_a_torn_trailing_line_is_dropped_not_fatal(tmp_path):
    """A process killed mid-write leaves half a line; that record must simply be redone."""
    path = tmp_path / "p.jsonl"
    path.write_text('{"offer_id": "a", "v": 1}\n{"offer_id": "b", "par', encoding="utf-8")
    assert set(resume.load_completed(path)) == {"a"}


def test_a_row_missing_a_required_key_is_not_done(tmp_path):
    """The backfill mechanism: an older row shape is outstanding work, not completed work."""
    path = tmp_path / "p.jsonl"
    resume.write_jsonl(path, [
        {"offer_id": "old", "valid": True},                     # pre-taxonomy row
        {"offer_id": "new", "valid": True, "failure": None, "raw": "{}"},
    ])
    done = resume.load_completed(path, required_keys=("failure", "raw"))
    assert set(done) == {"new"}
    assert set(resume.load_completed(path)) == {"old", "new"}, "without the requirement, both count"


def test_a_row_produced_under_other_settings_is_not_done(tmp_path):
    """Shape is not enough: a correctly-shaped row can still be the wrong measurement."""
    path = tmp_path / "p.jsonl"
    resume.write_jsonl(path, [
        {"offer_id": "old", "max_tokens": 1024},
        {"offer_id": "new", "max_tokens": 2048},
    ])
    done = resume.load_completed(path, matches=lambda r: r.get("max_tokens") == 2048)
    assert set(done) == {"new"}
    assert set(resume.load_completed(path)) == {"old", "new"}, "without the predicate, both count"


def test_required_keys_and_matches_both_have_to_pass(tmp_path):
    path = tmp_path / "p.jsonl"
    resume.write_jsonl(path, [
        {"offer_id": "shape-only", "max_tokens": 1024, "raw": "{}"},
        {"offer_id": "config-only", "max_tokens": 2048},
        {"offer_id": "both", "max_tokens": 2048, "raw": "{}"},
    ])
    done = resume.load_completed(
        path, required_keys=("raw",), matches=lambda r: r.get("max_tokens") == 2048,
    )
    assert set(done) == {"both"}


def test_append_sink_heals_a_torn_file_before_appending(tmp_path):
    """The rewrite is what stops an append from merging onto a half-written record."""
    path = tmp_path / "p.jsonl"
    path.write_text('{"offer_id": "a", "v": 1}\n{"offer_id": "b", "par', encoding="utf-8")
    done = resume.load_completed(path)
    with resume.append_sink(path, done) as sink:
        sink({"offer_id": "b", "v": 2})
    assert _rows(path) == [{"offer_id": "a", "v": 1}, {"offer_id": "b", "v": 2}]


def test_append_sink_keeps_the_completed_map_current(tmp_path):
    path = tmp_path / "p.jsonl"
    done = resume.load_completed(path)
    with resume.append_sink(path, done) as sink:
        sink({"offer_id": "a", "v": 1})
        assert done["a"] == {"offer_id": "a", "v": 1}, "visible before the context exits"
    assert set(done) == {"a"}


def test_records_survive_an_exception_mid_run(tmp_path):
    """A crash at record 2 must not cost record 1 — that is the whole point of appending."""
    path = tmp_path / "p.jsonl"
    done = resume.load_completed(path)
    with contextlib.suppress(KeyboardInterrupt), resume.append_sink(path, done) as sink:
        sink({"offer_id": "a", "v": 1})
        raise KeyboardInterrupt
    assert _rows(path) == [{"offer_id": "a", "v": 1}]
    assert set(resume.load_completed(path)) == {"a"}, "a rerun resumes from here"


def test_backup_once_copies_then_refuses_to_overwrite_itself(tmp_path):
    path = tmp_path / "p.jsonl"
    resume.write_jsonl(path, [{"offer_id": "a", "v": 1}])

    saved = resume.backup_once(path, ".pre-taxonomy")
    assert saved is not None and _rows(saved) == [{"offer_id": "a", "v": 1}]

    resume.write_jsonl(path, [{"offer_id": "a", "v": 2}])
    assert resume.backup_once(path, ".pre-taxonomy") is None, "a second run must not clobber it"
    assert _rows(saved) == [{"offer_id": "a", "v": 1}], "the original measurement is still there"


def test_backup_once_is_a_noop_when_there_is_nothing_to_save(tmp_path):
    assert resume.backup_once(tmp_path / "absent.jsonl", ".pre-taxonomy") is None

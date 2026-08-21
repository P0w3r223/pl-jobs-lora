"""Resumable JSONL runs: stop a long local job and finish it later without redoing the work.

CPU inference over the frozen set costs hours (the zero-shot GGUF variant runs ~68 s per posting,
so a full pass is most of an afternoon). A run that cannot be interrupted is a run that must be
scheduled around, and one that loses everything to a crash at record 130 is worse. So every
long-running producer here appends each record as it is produced and skips, on the next start, what
is already on disk.

Three properties this file is responsible for:

- **A torn trailing line never breaks a resume.** A process killed mid-write leaves a half-written
  final line. It is dropped and its record recomputed, and the file is rewritten clean from the
  parsed records *before* any append, so an append can never merge onto a partial line and both
  this reader and the plain ``json.loads`` readers elsewhere always see valid JSONL.
- **Only OS-unsynced records are lost.** Each append is flushed, so a kill costs at most the
  records the OS had not yet written.
- **A stale record shape is not mistaken for done.** ``required_keys`` lets a caller declare the
  fields a *current* record must carry; rows written by an older version of the producer are
  treated as outstanding instead of silently kept. This is what lets a schema change be backfilled
  by re-running rather than by hand-editing files.
- **A stale record *configuration* is not mistaken for done either.** ``matches`` extends the same
  idea from shape to content: a row measured under settings the run no longer uses is outstanding.
  Without it, changing a decoding knob would leave a file that silently mixes two configurations
  while reporting as one measurement.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path


def load_completed(
    path: Path, *, key: str = "offer_id", required_keys: tuple[str, ...] = (),
    matches: Callable[[dict], bool] | None = None,
) -> dict[str, dict]:
    """Records already on disk, keyed by ``key`` — the ones a resume may skip.

    A line that does not parse is dropped (the torn tail of a killed run). A record missing any of
    ``required_keys`` is dropped too: it was written by an older producer and re-running is the
    only way to bring it up to the current shape.

    ``matches`` rejects records that are shaped correctly but were produced under settings this run
    no longer uses. Re-running them is the only way to make the file one measurement rather than
    two overlaid.
    """
    if not path.exists():
        return {}
    done: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if key not in record or any(k not in record for k in required_keys):
            continue
        if matches is not None and not matches(record):
            continue
        done[record[key]] = record
    return done


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Rewrite ``path`` from ``rows``, always with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


@contextmanager
def append_sink(
    path: Path, completed: dict[str, dict], *, key: str = "offer_id",
) -> Iterator[Callable[[dict], None]]:
    """Yield a sink that appends each record to ``path`` and records it in ``completed``.

    The file is first rewritten from ``completed``, which is what drops a torn trailing line and
    guarantees the newline the append relies on. Pass the same dict that came from
    :func:`load_completed` — the sink keeps it current, so the caller can assemble its result from
    that dict whether a record was cached or just produced.
    """
    write_jsonl(path, list(completed.values()))
    with path.open("a", encoding="utf-8") as fh:
        def sink(record: dict) -> None:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            completed[record[key]] = record

        yield sink


def backup_once(path: Path, suffix: str) -> Path | None:
    """Copy ``path`` aside before it is rewritten, unless that copy already exists.

    Used when a resume is about to discard records under the *old* schema: they are regenerable,
    but regenerating them costs the hours this module exists to protect. Returns the backup path,
    or None when there was nothing to back up or a backup was already taken.
    """
    if not path.exists():
        return None
    target = path.with_suffix(path.suffix + suffix)
    if target.exists():
        return None
    target.write_bytes(path.read_bytes())
    return target

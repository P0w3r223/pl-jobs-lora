"""A retired platform, and everything that had not heard.

`ADR-0004` chose "free Colab/Kaggle" and its 2026-08-21 amendment records that Colab was
**abandoned**: the Bielik repo was gated for that account and kernel restarts wiped the
gitignored `data/processed/`. The notebook targets Kaggle now. The ADR is not the defect — it
is the model, and this repository's practice of recording a correction rather than quietly
making it.

**The defect is what the amendment's own sentence does to a later reader.** It says *"The
docstrings and README that said 'Colab' now say hosted GPU"*, which is **true**, and names the
two classes it fixed. Naming the classes you fixed reads, to the next reader, as naming all the
classes there are — and `0010` §4 B-4d found fifteen lines in neither class still saying Colab:
packaging, CI, the ignore file, the config's comments, the training requirements, and a step
heading inside another decision. Packaging and CI are exactly where a platform name is trusted
most and read last.

So the sweep is run by an instrument against the retired vocabulary, and never inferred from
the amendment's prose. Ported from `it-job-radar`'s `tests/test_decisions.py`, which is the
first instrument this portfolio built for this shape, with two changes the port needed:

- **Case-insensitively**, because `requirements-train.txt`:1 says `COLAB-ONLY` — the loudest
  line in the file — and `git grep -c 'Colab'` misses it. A retired vocabulary is exactly the
  thing a tree spells inconsistently.
- **Headings inside a decision document are swept**, though their bodies are not. A dated
  record is supposed to still read as it was written; `ADR-0006`:104's `## Colab Step 0` is a
  manual wearing an ADR's number, and a reader follows it.

The fifteen lines the row counted are not fifteen sites this guard rewrote: ten were in the
manuals and the heading, and five sit in `ADR-0003` and `ADR-0006` bodies, which are dated
record and stay as written. `docs/research/` is out for the same reason and carries the same
`Date:`/`Status:` header — which is why `f6-data-availability.md` still says the dataset is
frozen on HF Hub, a sentence B-4c corrected on the page, in the README and in `.gitignore`.

The rule is not *never mention it*: the retired name may appear where its retirement appears
with it, in the decision's own words.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADR = ROOT / "docs" / "decisions" / "0004-training-infra.md"
DECISIONS = ROOT / "docs" / "decisions"

#: How the retired platform is spelled. One marker, and the sweep lowercases both sides.
MARKERS = ("colab",)
#: How a line says it is quoting the retirement rather than restating the claim. Taken from
#: the ADR rather than invented, and asserted against it below, so this guard hands out no
#: exemption on its own authority.
RETIRED_BY = ("abandoned", "not colab")
#: And what it must name alongside one of those, so a stray "abandoned" about something else
#: cannot excuse a claim it has nothing to do with.
CITES = "adr-0004"
#: How close the retirement has to sit to the claim. Two lines each way covers a wrapped
#: sentence and a comment run, and deliberately not a whole TOML table — `it-job-radar`'s own
#: docstring records what a wider scope cost there.
NEARBY = 2


def _tracked(*patterns: str) -> list[str]:
    """Repository-relative paths git tracks. A machine without git fails rather than skips:
    a skip reads as a pass in pytest's summary, which is how a guard stops running quietly."""
    done = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", *patterns],
                          capture_output=True, text=True, encoding="utf-8", check=True)
    return [one for one in done.stdout.split("\0") if one]


def _text(path: Path) -> str:
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def _prose(lines: list[str]) -> str:
    """Lines joined as one sentence, with comment and markup prefixes dropped.

    A claim that wraps escapes a per-line search: `requirements-train.txt` breaks *"training
    runs on Colab/Kaggle while the whole repo"* across two lines, and `.github/workflows/ci.yml`
    breaks its sentence between `is a Colab` and the next comment line.
    """
    return " ".join(re.sub(r"^[\s#*/>!-]*", "", one).strip() for one in lines).lower()


def _stated(path: Path, lines: list[str] | None = None) -> list[str]:
    """Every line claiming the retired platform, minus the ones quoting its retirement."""
    lines = _text(path).split("\n") if lines is None else lines
    found = []
    for n in range(len(lines)):
        # **Detection is per line, and the report therefore names a line that holds the
        # marker.** Folded with the next line, as the port inherited it, this reported the line
        # *before* every match too — eight of nineteen entries held no marker at all, including
        # `python-version: "3.12"` under a heading saying the platform is named there.
        #
        # The fold is not merely noisy here, it is dead: `_prose` joins with a space, so a
        # one-word marker can never be split across a line break, and `it-job-radar`'s fold
        # exists because its markers are phrases — `the browser`, `browser-side`. Dropped
        # rather than kept as insurance, because a branch that cannot fire is a branch nobody
        # can test. **The window stays**, and it is what the wrapped-sentence cases in this
        # tree actually need: `requirements-train.txt` and `ci.yml` both put the retirement a
        # line or two from the claim.
        if not any(one in _prose([lines[n]]) for one in MARKERS):
            continue
        window = _prose(lines[max(0, n - NEARBY):n + NEARBY + 2])
        if CITES in window and any(one in window for one in RETIRED_BY):
            continue
        found.append(f"{path.relative_to(ROOT).as_posix()}:{n + 1}: {lines[n].strip()[:90]}")
    return found


def _manuals() -> list[Path]:
    """Where this repository speaks in the present tense about how it is run.

    `docs/decisions/` is excluded whole and it is not an oversight — a dated record is supposed
    to still read as it was written, and stripping that stops it being a record. Its headings
    are swept separately below.
    """
    roots = [
        ROOT / "README.md",
        ROOT / "CLAUDE.md",
        ROOT / "pyproject.toml",
        ROOT / ".gitignore",
        ROOT / "requirements-train.txt",
        ROOT / "configs" / "config.yaml",
        ROOT / ".github" / "workflows" / "ci.yml",
        ROOT / "notebooks" / "train_qlora.ipynb",
        # The published page is the most present-tense surface here and was missing from this
        # list, which made the docstring's *"`docs/decisions/` is excluded whole"* read as the
        # only exclusion there was. It names no platform today; adding it now costs one line
        # and stops the gap being discovered later.
        ROOT / "docs" / "index.html",
    ]
    absent = [p.name for p in roots if not p.is_file()]
    assert absent == [], f"the sweep names files that are not there: {absent}"
    sources = [ROOT / one for one in _tracked("src")]
    assert len(sources) > 20, f"the source sweep found only {len(sources)} files under src/"
    return roots + sources


def test_the_vocabulary_this_guard_sweeps_for_is_the_decisions_own():
    """Both registries answered against `ADR-0004`, and the premise under them.

    A marker the ADR never uses is a word someone here thought sounded retired; an exemption
    word it never uses is an excuse this file grants itself. And if the amendment goes, this
    guard should stop demanding — loudly — rather than enforce a decision the repository has
    reversed a second time.
    """
    adr = _text(ADR).lower()
    assert "status:" in adr and "amendment" in adr, (
        "ADR-0004 no longer records an amendment, so the platform below is not retired")
    unused = [one for one in MARKERS if one not in adr]
    assert unused == [], f"{unused} is swept for and ADR-0004 never says it"
    unused = [one for one in RETIRED_BY if one not in adr]
    assert unused == [], f"{unused} exempts a line and ADR-0004 never says it"
    assert CITES in adr, f"{CITES!r} is how a line cites this decision, and it does not"


def test_no_manual_names_the_platform_adr_0004_abandoned():
    stated = [one for path in _manuals() for one in _stated(path)]
    assert stated == [], (
        "Colab was abandoned on 2026-08-04 (ADR-0004's amendment) and is named as the "
        "platform in:\n" + "\n".join(stated))


def test_no_decision_heading_sends_a_reader_to_the_abandoned_platform():
    """Bodies are the record; headings are the manual.

    `ADR-0006`'s `## Colab Step 0` is a step a reader performs, printed in a document whose
    prose is deliberately frozen — so the sweep takes the headings and leaves everything else.

    **`ADR-0004` itself is outside this sweep, and that is the guard's stated limit.** Its
    title is *"Training on Colab, hosted MLflow, HF Hub artifacts"* and its amendment heading
    is *"the hosted GPU is Kaggle, not Colab"*: the two places the retired name has to appear,
    in the one document whose whole subject is that it appeared. The cost is real and small —
    a step heading added to that file would escape this guard — and it is written here rather
    than left for a reader to discover, because a guard whose scope is unwritten is read as
    total.
    """
    stated = []
    for name in sorted(_tracked("docs/decisions")):
        path = ROOT / name
        if path == ADR:
            continue
        headings = [one if one.lstrip().startswith("#") else "" for one in _text(path).split("\n")]
        stated += _stated(path, headings)
    assert stated == [], (
        "a heading tells a reader to work on the platform ADR-0004 abandoned:\n"
        + "\n".join(stated))

"""The README is a surface too, and until now nothing read it.

`docs/index.html` has 327 lines of `tests/test_docs_page.py` over it, enforcing `0007` §5.0 —
*every figure a surface prints is a figure a committed artifact prints*. The README reproduces
the same evaluation table and was read by nothing: `git grep README` across `tests/` was empty.

**The gap is not theoretical and the repository has already paid for it once.** `305f4e2` is
titled *"the README is a surface too, and the guard was stricter than the spec"*; it went
through that table by hand, corrected four cells in two rows — `21.4`→`21.43`, `103.0`→`103.02`,
`2.3`→`2.25`, `4.1`→`4.13` — and left three in the other two rows. A hand pass over a table is a
hand count, and this portfolio's hand counts have read 15, 18 and 19 against a true 20.
`0010` §4 B-4a is the row; the three it left were `102.6`, `2.5` and `4.7` against the
artifact's `102.58`, `2.47` and `4.68`.

**The model keys come out of both sides, and that is the whole reason this guard is not the
page's.** `configs/config.yaml` spells a candidate `qwen2.5-1.5b`, so a naive sweep reads a
bare `2.5` on the source side and accepts a rounded latency of `2.5` s — one of the three
defects above, sourced by a coincidence of spelling. The keys are removed from both sides, and
they are read from the config rather than listed here, so a new candidate cannot quietly widen
what this test admits.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from test_docs_page import _NUMBER, _SEPARATORS, _canonical

from pl_jobs_lora.config import load_config

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

#: Fenced blocks are commands and output a reader retypes, not claims the README makes, and
#: markdown link targets carry digits that belong to a URL. Both come out before tokenising —
#: the same reduction `0010` §4 B-4a measured the README with.
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_LINK = re.compile(r"\]\([^)]*\)")


def _tracked(*patterns: str) -> list[str]:
    done = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", *patterns],
                          capture_output=True, text=True, encoding="utf-8", check=True)
    return [one for one in done.stdout.split("\0") if one]


def _without_model_keys(text: str) -> str:
    """`qwen2.5-1.5b` is a name, not three figures. Read from the config, never listed here.

    **Case-insensitively**, and that is not tidiness: the config spells the key `qwen2.5-1.5b`
    while `CLAUDE.md`, `ADR-0001` and the page all write `Qwen2.5-1.5B`, which is the vendor's
    spelling. An exact replace leaves those three intact, and each of them then donates a bare
    `2.5` to the source side — enough to source a rounded latency cell of `2.5` s, which is one
    of the three defects this file exists to catch. Found by the mutation battery, which put
    the cell back and watched this guard stay green.
    """
    for candidate in load_config().models:
        for spelling in (candidate.key, candidate.hf_repo, candidate.gguf_repo,
                         candidate.gguf_file):
            if spelling:
                text = re.sub(re.escape(spelling), " ", text, flags=re.IGNORECASE)
    return text


def _figures(text: str) -> set[str]:
    return {_canonical(token.translate(_SEPARATORS))
            for token in re.findall(_NUMBER, _without_model_keys(text))}


def _readme_claims() -> set[str]:
    body = _LINK.sub("]", _FENCE.sub(" ", README.read_text(encoding="utf-8")))
    return _figures(body)


def _sourced() -> set[str]:
    """Every figure any other tracked text file prints.

    The corpus is `git ls-files` rather than a walk: a filesystem walk sees
    `data/processed/manifest.json` and `results/eval/predictions/`, which `.gitignore` keeps
    out of every clone — and sourcing a README figure to a file no reader has is the defect
    `0010` §4 B-4b raises two paragraphs further down the same row.
    """
    out: set[str] = set()
    for name in _tracked():
        path = ROOT / name
        # **`tests/` is not a source**, and this file is the reason the rule is written down.
        # Its own docstring quotes `102.6`, `2.5` and `4.7` — the three figures it refuses — so
        # with `tests/` in the corpus the guard sourced them to itself and went green over the
        # defect it was written for. The index records the same shape in `tools/citations.py`:
        # prose about a broken citation may not write one.
        if name.startswith("tests/") or path == README:
            continue
        if path.suffix in {".png", ".jpg", ".svg", ".ico", ".woff2"}:
            continue
        try:
            out |= _figures(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
    return out


def test_the_model_key_exemption_is_load_bearing_and_not_decoration():
    """A bare `2.5` reaches the source side only through a model name, and must not survive.

    The exemption is the one part of this guard a mutation cannot reach from outside: remove
    it and nothing turns red until a README cell happens to round to a model's version number.
    So it is asserted directly — the tree donates `2.5`, and it stops donating it once the
    candidates' spellings come out.
    """
    tree = " ".join((ROOT / name).read_text(encoding="utf-8", errors="ignore")
                    for name in _tracked("docs", "CLAUDE.md", "configs"))
    raw = {_canonical(token.translate(_SEPARATORS)) for token in re.findall(_NUMBER, tree)}
    assert "2.5" in raw, "the tree no longer spells a model version this guard has to exempt"
    assert "2.5" not in _figures(tree), (
        "a model name is donating a bare figure to the source side, so a README cell rounding "
        "to it would read as sourced")


def test_every_figure_the_readme_prints_is_one_a_committed_artifact_prints():
    """`0007` §5.0, applied to the second surface this repository publishes.

    **This is a floor and not a proof**, for the same reason `test_docs_page.py` says so of
    itself: the corpus is every tracked text file, which is wide, so a figure can be sourced
    by a coincidence rather than by the artifact that means it. What it does catch is the
    class it was written for — a number that appears *nowhere*, which is what a rounded cell
    is.
    """
    unsourced = sorted(_readme_claims() - _sourced())
    assert unsourced == [], (
        "the README prints figures no committed artifact prints: " + ", ".join(unsourced)
        + "\nA rounded cell is a new number, and no reader can check it against anything.")

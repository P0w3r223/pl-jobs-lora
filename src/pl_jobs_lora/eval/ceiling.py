"""How much of a label is answerable from the input at all (ADR-0003): the data ceiling.

Pure and offline. No model, no network, no predictions — this measures the **dataset**, by asking
of every gold term whether it occurs anywhere in the prose that model is given.

Why an F1 needs this next to it. The gold tech labels come from theprotocol's structured
technologies widget, and ADR-0002's leakage guard deliberately strips that widget out of the prose
— otherwise the task would be copying a list rather than reading a posting. The consequence went
unquantified until 2026-08-21: only about a third of `tech_expected` terms and a seventh of
`tech_optional` terms appear in the text at all. A model cannot extract what is not there, so a
raw F1 on these fields conflates "the model missed it" with "it was never in the input".

Reported beside the score, `0.28` on `tech_expected` stops reading as *the model is weak* and
starts reading as *the model recovered most of what was recoverable*.

**This is a bound, not a target.** Presence of a term is necessary for extraction, not sufficient,
so the true achievable F1 is at or below this number. It is also approximate in the conservative
direction: matching uses the vendored alias map, and a canonical term with no alias entry falls
back to exact word-boundary matching, so an unusual surface form is counted as absent. The number
is therefore best read as a **floor on the ceiling**, and comparisons *between* fields — measured
the same way — carry more weight than any single value.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

# Only open-vocabulary term lists get a ceiling. `seniority` and `work_mode` are closed vocabularies
# expressed in free Polish prose ("starszy", "zdalnie", "praca hybrydowa"), where the absence of a
# canonical token says nothing about whether the fact is stated — a term-presence test there would
# measure the vocabulary, not the data.
CEILING_FIELDS = ("tech_expected", "tech_optional")


@dataclass
class FieldCeiling:
    """Per-field answerability of the gold labels against the prose the model actually sees."""

    field: str
    terms: int                 # gold terms across the scored records
    present: int               # of those, how many occur in their own record's prose
    records_with_gold: int     # records whose gold carries at least one term
    records_unanswerable: int  # ...of which none of the terms occur in the prose

    @property
    def share(self) -> float | None:
        """Upper bound on recall: the fraction of gold terms that are in the input at all."""
        return self.present / self.terms if self.terms else None

    @property
    def unanswerable_share(self) -> float | None:
        return (
            self.records_unanswerable / self.records_with_gold
            if self.records_with_gold else None
        )

    def as_dict(self) -> dict:
        return {
            "field": self.field, "terms": self.terms, "present": self.present,
            "records_with_gold": self.records_with_gold,
            "records_unanswerable": self.records_unanswerable,
            "share": None if self.share is None else round(self.share, 4),
            "unanswerable_share": (
                None if self.unanswerable_share is None else round(self.unanswerable_share, 4)
            ),
        }


def _surface_forms(alias_index: dict[str, str]) -> dict[str, set[str]]:
    """Canonical term -> every spelling that normalizes to it, plus the canonical form itself."""
    forms: dict[str, set[str]] = defaultdict(set)
    for alias, canon in alias_index.items():
        forms[canon].add(alias.lower())
        forms[canon].add(canon.lower())
    return forms


def _compile(forms: dict[str, set[str]]) -> dict[str, re.Pattern]:
    # Longest-first so "react native" wins over "react" when both are spellings of one term.
    return {
        canon: re.compile(
            "|".join(rf"(?<!\w){re.escape(s)}(?!\w)" for s in sorted(spellings, key=len,
                                                                     reverse=True)),
            re.IGNORECASE,
        )
        for canon, spellings in forms.items()
    }


def term_in_prose(term: str, prose: str, patterns: dict[str, re.Pattern]) -> bool:
    """Whether a canonical gold term (in any known spelling) occurs in this posting's prose."""
    pattern = patterns.get(term)
    if pattern is not None:
        return bool(pattern.search(prose))
    # No alias entry: fall back to the canonical form alone. Conservative — an unusual spelling
    # reads as absent, which understates the ceiling rather than inflating it.
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", prose, re.IGNORECASE) is not None


def answerable_ceilings(
    records: list[dict], alias_index: dict[str, str],
    fields: tuple[str, ...] = CEILING_FIELDS,
) -> dict[str, FieldCeiling]:
    """Per-field ceiling over records shaped ``{"prose": str, "gold": {field: [terms]}}``."""
    patterns = _compile(_surface_forms(alias_index))
    out: dict[str, FieldCeiling] = {}
    for field in fields:
        terms = present = with_gold = unanswerable = 0
        for record in records:
            prose = record.get("prose") or ""
            gold_terms = (record.get("gold") or {}).get(field) or []
            if not gold_terms:
                continue
            with_gold += 1
            hits = sum(term_in_prose(t, prose, patterns) for t in gold_terms)
            terms += len(gold_terms)
            present += hits
            if hits == 0:
                unanswerable += 1
        out[field] = FieldCeiling(
            field=field, terms=terms, present=present,
            records_with_gold=with_gold, records_unanswerable=unanswerable,
        )
    return out


def render_markdown(ceilings: dict[str, FieldCeiling], recalls: dict[str, float | None]) -> str:
    """The data-ceiling section: what was answerable, and how much of that each variant got."""
    from pl_jobs_lora.eval.scoring import fmt_metric

    lines = [
        "## Data ceiling (model-free)\n",
        "Gold tech labels come from the platform's technologies widget, which the ADR-0002 "
        "leakage guard strips out of the prose — otherwise the task would be copying a list. "
        "This measures the consequence: the share of gold terms that appear in the posting text "
        "**at all**. It bounds recall from above, so a raw F1 on these fields cannot be read "
        "without it. Necessary, not sufficient — and matching is conservative, so treat it as a "
        "floor on the ceiling.\n",
        "| field | gold terms | in prose | ceiling on recall | postings with gold | "
        "...where no term is in the prose |",
        "|---|---|---|---|---|---|",
    ]
    for c in ceilings.values():
        lines.append(
            f"| {c.field} | {c.terms} | {c.present} | {fmt_metric(c.share)} | "
            f"{c.records_with_gold} | {c.records_unanswerable} "
            f"({fmt_metric(c.unanswerable_share)}) |"
        )
    measured = {f: r for f, r in recalls.items() if r is not None and ceilings.get(f)}
    if measured:
        lines += ["", "Best recall achieved, against that ceiling:", ""]
        for f, recall in measured.items():
            share = ceilings[f].share
            reached = f"{recall / share:.0%}" if share else "-"
            lines.append(
                f"- `{f}`: {fmt_metric(recall)} of a ceiling of {fmt_metric(share)} "
                f"— **{reached} of what the input makes recoverable**."
            )
    return "\n".join(lines)

"""Pure, model-free label-agreement layer (S3; ADR-0005): label sources -> agreement.

Never loads a model and does no network I/O — the triangulated QA math only. Each *leg* is a
list of label records in the gold shape ``{offer_id, title, seniority, work_mode, tech_expected,
tech_optional, salary}``; the LLM leg may carry ``valid`` (JSON-validity, defaults True). All three
sides are already normalized upstream (platform gold via the collector, LLM/arbiter output via the
prompt parser), so agreement is set/scalar comparison — order-invariant and alias-fair.

Three legs feed the report: LLM (Bielik-1.5B few-shot, prose-only) vs platform gold at full
scale, and — on the human-checked sample — LLM vs human and platform vs human, plus a per-field
triangulation bucketing each field into {all agree / platform gap caught / LLM error / all differ}.
Beyond the reused per-field F1 (``eval.scoring``), it adds a raw agreement rate and Cohen's kappa
for the categorical fields, so "the labels are trustworthy" is chance-corrected, not just raw.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable
from dataclasses import dataclass, field

from pl_jobs_lora.eval.scoring import SET_FIELDS, _score_salary, score_predictions

# Categorical fields whose agreement is chance-corrected with Cohen's kappa. The tech fields are an
# open vocabulary — kappa-as-single-category is uninformative there — so only these two get it.
CATEGORICAL_KAPPA_FIELDS = ("seniority", "work_mode")
# Every field the triangulation classifies (the four set fields + salary + title).
TRIANGULATION_FIELDS = (*SET_FIELDS, "salary", "title")
# Human is the reference: le = LLM==human, pe = platform==human.
BUCKETS = ("all_agree", "platform_gap_caught", "llm_error", "all_differ")


def _by_id(records: list[dict]) -> dict[str, dict]:
    return {r["offer_id"]: r for r in records}


def _canon(field_name: str, value) -> Hashable:
    """A hashable canonical category for a field value (used for Cohen's kappa)."""
    if field_name == "title":
        return (value or "").strip().lower()
    if field_name == "salary":
        v = value or {}
        return (v.get("kind"), v.get("currency"), v.get("amount_from"), v.get("amount_to"))
    return frozenset(value or [])  # set fields: normalized set as one category


def _field_equal(field_name: str, a, b, *, salary_rel_tolerance: float) -> bool:
    """Field-level equality, reusing the scorer's salary tolerance so it matches ADR-0003."""
    if field_name == "salary":
        s = _score_salary(a, b, salary_rel_tolerance)
        return s["currency"] == 1 and s["kind"] == 1 and s["amount"] == 1
    if field_name == "title":
        return (a or "").strip().lower() == (b or "").strip().lower()
    return frozenset(a or []) == frozenset(b or [])


def cohen_kappa(a_labels: list[Hashable], b_labels: list[Hashable]) -> float:
    """Chance-corrected agreement over paired categorical labels; guards the pe==1 case."""
    n = len(a_labels)
    if n == 0:
        return 1.0
    po = sum(a == b for a, b in zip(a_labels, b_labels, strict=True)) / n
    ca, cb = Counter(a_labels), Counter(b_labels)
    pe = sum((ca[c] / n) * (cb.get(c, 0) / n) for c in ca)
    if pe >= 1.0:  # only one category present on a side → agreement is entirely by chance
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


def raw_agreement(
    a: list[dict], b: list[dict], field_name: str, *, salary_rel_tolerance: float,
) -> float:
    """Fraction of the matched population where ``field_name`` is equal on both sides."""
    a_by, b_by = _by_id(a), _by_id(b)
    ids = [i for i in b_by if i in a_by]
    if not ids:
        return 1.0
    hits = sum(
        _field_equal(field_name, a_by[i].get(field_name), b_by[i].get(field_name),
                     salary_rel_tolerance=salary_rel_tolerance)
        for i in ids
    )
    return hits / len(ids)


@dataclass
class PairwiseReport:
    """One label-source pair: reused F1/exact + raw agreement + kappa (categorical) + salary."""

    n: int
    json_validity: float
    title_agreement: float
    title_exact: float
    per_field: dict = field(default_factory=dict)
    salary: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "json_validity": round(self.json_validity, 4),
            "title_agreement": round(self.title_agreement, 4),
            "title_exact": round(self.title_exact, 4),
            "per_field": self.per_field,
            "salary": self.salary,
        }


def pairwise(a: list[dict], b: list[dict], *, salary_rel_tolerance: float) -> PairwiseReport:
    """Agreement between two label sources; ``a`` is the proposed side (carries ``valid``)."""
    a_by, b_by = _by_id(a), _by_id(b)
    ids = [i for i in b_by if i in a_by]
    predictions = [
        {"offer_id": i, "valid": a_by[i].get("valid", True), "parsed": a_by[i]} for i in ids
    ]
    gold = [b_by[i] for i in ids]
    score = score_predictions(predictions, gold, salary_rel_tolerance=salary_rel_tolerance)

    per_field: dict = {}
    for f in SET_FIELDS:
        agreement = raw_agreement(a, b, f, salary_rel_tolerance=salary_rel_tolerance)
        entry = {
            "agreement": round(agreement, 4),
            "f1": score.fields[f]["f1"],
            "exact_match": score.fields[f]["exact_match"],
        }
        if f in CATEGORICAL_KAPPA_FIELDS:
            a_lab = [_canon(f, a_by[i].get(f)) for i in ids]
            b_lab = [_canon(f, b_by[i].get(f)) for i in ids]
            entry["kappa"] = round(cohen_kappa(a_lab, b_lab), 4)
        per_field[f] = entry

    return PairwiseReport(
        n=score.n,
        json_validity=score.json_validity,
        title_agreement=raw_agreement(a, b, "title", salary_rel_tolerance=salary_rel_tolerance),
        title_exact=score.title_exact,
        per_field=per_field,
        salary=score.salary,
    )


@dataclass
class TriangulationReport:
    """Per-field bucket counts over the human-checked sample (human is the reference)."""

    n: int
    per_field: dict = field(default_factory=dict)
    totals: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"n": self.n, "per_field": self.per_field, "totals": self.totals}


def triangulate(
    llm: list[dict], platform: list[dict], human: list[dict], *, salary_rel_tolerance: float,
) -> TriangulationReport:
    """Classify each field of each human-checked record into one of the four buckets."""
    l_by, p_by, h_by = _by_id(llm), _by_id(platform), _by_id(human)
    ids = list(h_by)  # the sample population is exactly the human-adjudicated records
    per_field = {f: dict.fromkeys(BUCKETS, 0) for f in TRIANGULATION_FIELDS}
    for i in ids:
        lr, pr, hr = l_by.get(i, {}), p_by.get(i, {}), h_by[i]
        for f in TRIANGULATION_FIELDS:
            le = _field_equal(f, lr.get(f), hr.get(f), salary_rel_tolerance=salary_rel_tolerance)
            pe = _field_equal(f, pr.get(f), hr.get(f), salary_rel_tolerance=salary_rel_tolerance)
            if le and pe:
                bucket = "all_agree"
            elif le and not pe:
                bucket = "platform_gap_caught"  # LLM==human != platform → platform label wrong
            elif pe and not le:
                bucket = "llm_error"  # platform==human != LLM → the LLM misread the prose
            else:
                bucket = "all_differ"
            per_field[f][bucket] += 1
    totals = {b: sum(per_field[f][b] for f in TRIANGULATION_FIELDS) for b in BUCKETS}
    return TriangulationReport(n=len(ids), per_field=per_field, totals=totals)


@dataclass
class AgreementReport:
    """The full S3 report: the automatic leg + the two human legs + triangulation. Numbers only."""

    metadata: dict
    llm_vs_platform: PairwiseReport
    llm_vs_human: PairwiseReport | None = None
    platform_vs_human: PairwiseReport | None = None
    triangulation: TriangulationReport | None = None

    def as_dict(self) -> dict:
        return {
            "metadata": self.metadata,
            "llm_vs_platform": self.llm_vs_platform.as_dict(),
            "llm_vs_human": self.llm_vs_human.as_dict() if self.llm_vs_human else None,
            "platform_vs_human": (
                self.platform_vs_human.as_dict() if self.platform_vs_human else None
            ),
            "triangulation": self.triangulation.as_dict() if self.triangulation else None,
        }

    def render_markdown(self) -> str:
        lines = ["# Labeling-QA agreement report", ""]
        for k, v in self.metadata.items():
            lines.append(f"- **{k}**: {v}")
        lines += ["", "## LLM ↔ platform (full scale)", "", _pairwise_md(self.llm_vs_platform)]
        if self.llm_vs_human:
            lines += ["", "## LLM ↔ human (sample)", "", _pairwise_md(self.llm_vs_human)]
        if self.platform_vs_human:
            lines += ["", "## Platform ↔ human (sample)", "", _pairwise_md(self.platform_vs_human)]
        if self.triangulation:
            lines += ["", "## Triangulation (per field, human = reference)", "",
                      _triangulation_md(self.triangulation)]
        return "\n".join(lines) + "\n"


def _pairwise_md(r: PairwiseReport) -> str:
    rows = [
        f"- n = {r.n}, JSON-validity = {round(r.json_validity, 4)}, "
        f"title agreement = {round(r.title_agreement, 4)} (exact = {round(r.title_exact, 4)})",
        "",
        "| field | agreement | F1 | exact | kappa |",
        "|---|---|---|---|---|",
    ]
    for f, m in r.per_field.items():
        rows.append(
            f"| {f} | {m['agreement']} | {m['f1']} | {m['exact_match']} | {m.get('kappa', 'n/a')} |"
        )
    rows.append(f"| salary | {r.salary} | | | |")
    return "\n".join(rows)


def _triangulation_md(t: TriangulationReport) -> str:
    rows = [f"n = {t.n}", "", "| field | " + " | ".join(BUCKETS) + " |",
            "|---|" + "---|" * len(BUCKETS)]
    for f, counts in t.per_field.items():
        rows.append(f"| {f} | " + " | ".join(str(counts[b]) for b in BUCKETS) + " |")
    rows.append("| **total** | " + " | ".join(str(t.totals[b]) for b in BUCKETS) + " |")
    return "\n".join(rows)

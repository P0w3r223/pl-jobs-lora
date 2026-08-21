"""Pure, model-free per-field scorer (ADR-0003): (predictions, gold) -> scores.

Never loads a model and does no network I/O. Both sides are already normalized upstream (the
collector builds gold via the vendored maps, the probe parser normalizes model output), so
scoring is set/scalar comparison only — order-invariant and alias-fair.

Two properties the metrics must have, and which an earlier version of this module did not:

- **Silence earns nothing.** Every field reports ``support`` — the number of records whose *gold*
  carries that field — and value accuracy is denominated by ``support``, never by the record
  count. About 69 % of gold records carry no salary, so a scorer that credits ``None == None``
  hands a model emitting nothing at all ~0.75 on every salary sub-metric. Presence/absence is
  scored separately as ``detection``, where correctly deciding "absent" *is* the right answer.
- **An invalid prediction scores zero on every field**, as this docstring has always claimed. It
  is treated as no answer given: no true positives, no exact match, and a failed detection.

A metric whose ``support`` is zero is reported as ``None`` — unmeasurable on this test set —
rather than as a perfect score.
"""

from __future__ import annotations

from dataclasses import dataclass

SET_FIELDS = ("seniority", "work_mode", "tech_expected", "tech_optional")

# Fields the headline `field F1` averages. `tech_optional` is scored and reported like any other
# field but deliberately excluded here: only 14-17 % of its gold terms occur anywhere in the prose
# the model is given (the platform files them in the technologies widget, which the ADR-0002
# leakage guard strips), and 75 % of the postings that carry gold optional terms contain none of
# them. Averaging a label the input does not contain into the headline measures the dataset, not
# the model. See ADR-0003's 2026-08-21 amendment and `eval.ceiling`.
HEADLINE_FIELDS = ("seniority", "work_mode", "tech_expected")

# Why a prediction was not usable — the vocabulary the parser writes and the report counts.
# It lives in this module (not in ``eval.prompt``, which owns the parsing) so the offline report
# can tabulate failures without importing the prompt's HTTP-carrying dependency chain.
EMPTY_OUTPUT = "empty_output"            # the model returned nothing at all
NO_JSON_OBJECT = "no_json_object"        # prose only — no "{" anywhere in the output
JSON_DECODE_ERROR = "json_decode_error"  # a "{" that does not close: usually a hit token cap
SCHEMA_INVALID = "schema_invalid"        # an object the JobPosting contract rejects
# Rows written before the taxonomy existed: invalid, but the reason was never recorded. Counted
# under its own name so an old predictions file cannot masquerade as a clean one.
UNRECORDED = "unrecorded"

# No "not an object" class: the parser scans to the first "{" and decodes from there, and a decode
# anchored on "{" yields a dict or raises. A class that cannot occur has no place in a published
# taxonomy — it would read as "this never happens" when it means "this cannot be detected".
PARSE_FAILURES = (
    EMPTY_OUTPUT, NO_JSON_OBJECT, JSON_DECODE_ERROR, SCHEMA_INVALID, UNRECORDED,
)


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def mean_measured_f1(
    fields: dict, headline_fields: tuple[str, ...] = HEADLINE_FIELDS,
) -> float | None:
    """Headline field F1: the mean over ``headline_fields`` that *have* support on this test set.

    Two exclusions, for two different reasons. A field with no support is unmeasurable, not zero —
    averaging it in would drag the headline toward whichever fields the test set happens not to
    exercise. A field outside ``headline_fields`` is excluded by decision, not by accident: it is
    still scored and reported, but is not treated as evidence about the model. ``None`` when
    nothing was measurable at all.
    """
    measured = [
        fields[f]["f1"] for f in headline_fields
        if f in fields and fields[f]["f1"] is not None
    ]
    return sum(measured) / len(measured) if measured else None


def fmt_metric(value: float | None, digits: int = 2) -> str:
    """Render a metric for a table: ``-`` marks *unmeasurable on this test set*, never a top score.

    Lives here rather than in each renderer because this module is what introduced the ``None``:
    every table that displays these metrics owes the reader the same convention.
    """
    return "-" if value is None else f"{value:.{digits}f}"


@dataclass
class SetMetric:
    """Micro-averaged P/R/F1 over set-valued fields, plus exact match over supported records."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    exact_supported: int = 0    # exact matches among records whose gold is non-empty
    n: int = 0                  # records scored
    support: int = 0            # records whose gold carries the field
    n_pred_nonempty: int = 0    # records where the model predicted anything at all

    def add(self, pred: list | None, gold: list | None) -> None:
        ps, gs = set(pred or []), set(gold or [])
        self.tp += len(ps & gs)
        self.fp += len(ps - gs)
        self.fn += len(gs - ps)
        self.n += 1
        if gs:
            self.support += 1
            self.exact_supported += int(ps == gs)
        if ps:
            self.n_pred_nonempty += 1

    @property
    def measurable(self) -> bool:
        """False when neither side ever carried the field — nothing to score, not a top score."""
        return self.support > 0 or self.n_pred_nonempty > 0

    @property
    def precision(self) -> float | None:
        if not self.measurable:
            return None
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float | None:
        if not self.measurable:
            return None
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None:
            return None
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def exact_match(self) -> float | None:
        """Over records whose gold is non-empty — empty-vs-empty is not an achievement."""
        return self.exact_supported / self.support if self.support else None

    def as_dict(self) -> dict:
        return {
            "precision": _round(self.precision), "recall": _round(self.recall),
            "f1": _round(self.f1), "exact_match": _round(self.exact_match),
            "n": self.n, "support": self.support, "n_pred_nonempty": self.n_pred_nonempty,
        }


@dataclass
class SalaryMetric:
    """Detection (present/absent, over all records) split from value accuracy (over support)."""

    n: int = 0
    support: int = 0        # records whose gold carries a salary
    detected: int = 0       # present/absent decision correct
    currency: int = 0
    kind: int = 0
    amount: int = 0

    def add(self, pred: dict | None, gold: dict | None, rel_tol: float, *, valid: bool) -> None:
        self.n += 1
        gold_present = gold is not None
        if gold_present:
            self.support += 1
        if not valid:
            return  # no answer given: no detection credit and no value credit
        pred_present = pred is not None
        self.detected += int(gold_present == pred_present)
        if not (gold_present and pred_present):
            return
        self.currency += int(pred.get("currency") == gold.get("currency"))
        self.kind += int(pred.get("kind") == gold.get("kind"))
        self.amount += int(
            _amount_ok(pred.get("amount_from"), gold.get("amount_from"), rel_tol)
            and _amount_ok(pred.get("amount_to"), gold.get("amount_to"), rel_tol)
        )

    def _over_support(self, hits: int) -> float | None:
        return hits / self.support if self.support else None

    def as_dict(self) -> dict:
        return {
            "detection": _round(self.detected / self.n if self.n else None),
            "currency": _round(self._over_support(self.currency)),
            "kind": _round(self._over_support(self.kind)),
            "amount": _round(self._over_support(self.amount)),
            "n": self.n, "support": self.support,
        }


def _amount_ok(pred: float | None, gold: float | None, rel_tol: float) -> bool:
    """Bounds match within a relative tolerance band; a missing bound must match a missing one."""
    if gold is None:
        return pred is None
    if pred is None:
        return False
    return abs(pred - gold) <= rel_tol * abs(gold)


def salary_equal(a: dict | None, b: dict | None, rel_tol: float) -> bool:
    """Whether two *annotations* of a posting agree on salary — both absent counts as agreement.

    Deliberately distinct from :class:`SalaryMetric`, which scores a prediction against gold: there
    "both absent" is a detection success and earns no value credit, because a model that emits
    nothing would otherwise inherit the ~69 % of records that carry no salary. Inter-annotator
    agreement has no such asymmetry — two annotators who both read "no salary" genuinely agree.
    """
    if a is None or b is None:
        return a is None and b is None
    return (
        a.get("currency") == b.get("currency")
        and a.get("kind") == b.get("kind")
        and _amount_ok(a.get("amount_from"), b.get("amount_from"), rel_tol)
        and _amount_ok(a.get("amount_to"), b.get("amount_to"), rel_tol)
    )


@dataclass
class ScoreReport:
    n: int                          # prediction rows matched to gold and scored
    n_gold: int
    coverage: float                 # distinct gold records answered — a partial run must show
    n_duplicate_rows: int           # rows answering an id already answered; they double-weight it
    json_validity: float
    title_exact: float | None
    title_support: int
    fields: dict
    salary: dict

    def as_dict(self) -> dict:
        return {
            "n": self.n, "n_gold": self.n_gold, "coverage": round(self.coverage, 4),
            "n_duplicate_rows": self.n_duplicate_rows,
            "json_validity": round(self.json_validity, 4),
            "title_exact": _round(self.title_exact), "title_support": self.title_support,
            "fields": self.fields, "salary": self.salary,
        }


def score_predictions(
    predictions: list[dict], gold: list[dict], *, salary_rel_tolerance: float,
) -> ScoreReport:
    """Aggregate per-field scores. predictions: {offer_id, valid, parsed}; gold: {offer_id, ...}."""
    gold_by_id = {g["offer_id"]: g for g in gold}
    metrics = {f: SetMetric() for f in SET_FIELDS}
    salary = SalaryMetric()
    valid_count = title_hits = title_support = matched = 0
    # Coverage is denominated by *distinct* gold records answered, so a predictions file that
    # appended a record twice (the resume path in labeling_qa writes that way) cannot report
    # more than complete coverage of a set it only partly answered.
    matched_ids: set[str] = set()

    for pred in predictions:
        g = gold_by_id.get(pred["offer_id"])
        if g is None:
            continue
        matched += 1
        matched_ids.add(pred["offer_id"])
        valid = bool(pred.get("valid", False))
        valid_count += int(valid)
        # An unparseable prediction is scored as no answer at all, never as an empty match.
        parsed = (pred.get("parsed") or {}) if valid else {}

        for f in SET_FIELDS:
            metrics[f].add(parsed.get(f), g.get(f))
        salary.add(parsed.get("salary"), g.get("salary"), salary_rel_tolerance, valid=valid)

        gold_title = (g.get("title") or "").strip().lower()
        if gold_title:
            title_support += 1
            title_hits += int((parsed.get("title") or "").strip().lower() == gold_title)

    denom = matched or 1
    return ScoreReport(
        n=matched,
        n_gold=len(gold),
        coverage=len(matched_ids) / len(gold) if gold else 0.0,
        # Full coverage with duplicates is the one way the metrics can still mislead: the repeated
        # record is weighted twice in tp/fp/fn while coverage reads 1.00. Counted, not silent.
        n_duplicate_rows=matched - len(matched_ids),
        json_validity=valid_count / denom,
        title_exact=title_hits / title_support if title_support else None,
        title_support=title_support,
        fields={f: m.as_dict() for f, m in metrics.items()},
        salary=salary.as_dict(),
    )

"""Pure, model-free per-field scorer (ADR-0003): (predictions, gold) -> scores.

Never loads a model and does no network I/O. Both sides are already normalized upstream (the
collector builds gold via the vendored maps, the probe parser normalizes model output), so
scoring is set/scalar comparison only — order-invariant and alias-fair.

A prediction that failed to parse (``valid=False`` / ``parsed=None``) scores 0 on every field,
which is exactly how invalid JSON is penalized as a first-class metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SET_FIELDS = ("seniority", "work_mode", "tech_expected", "tech_optional")


@dataclass
class SetMetric:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    exact: int = 0
    n: int = 0

    def add(self, pred: list | None, gold: list | None) -> None:
        ps, gs = set(pred or []), set(gold or [])
        self.tp += len(ps & gs)
        self.fp += len(ps - gs)
        self.fn += len(gs - ps)
        self.exact += int(ps == gs)
        self.n += 1

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def exact_match(self) -> float:
        return self.exact / self.n if self.n else 0.0

    def as_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4), "recall": round(self.recall, 4),
            "f1": round(self.f1, 4), "exact_match": round(self.exact_match, 4), "n": self.n,
        }


def _score_salary(pred: dict | None, gold: dict | None, rel_tol: float) -> dict:
    """currency/kind exact, amount bounds within a relative tolerance band; None==None matches."""
    def _amount_ok(p, g) -> bool:
        if g is None:
            return p is None
        if p is None:
            return False
        return abs(p - g) <= rel_tol * abs(g)

    if pred is None and gold is None:
        return {"currency": 1, "kind": 1, "amount": 1}
    pred, gold = pred or {}, gold or {}
    amount = int(
        _amount_ok(pred.get("amount_from"), gold.get("amount_from"))
        and _amount_ok(pred.get("amount_to"), gold.get("amount_to"))
    )
    return {
        "currency": int(pred.get("currency") == gold.get("currency")),
        "kind": int(pred.get("kind") == gold.get("kind")),
        "amount": amount,
    }


@dataclass
class ScoreReport:
    n: int
    json_validity: float
    title_exact: float
    fields: dict = field(default_factory=dict)
    salary: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n": self.n, "json_validity": round(self.json_validity, 4),
            "title_exact": round(self.title_exact, 4),
            "fields": self.fields, "salary": self.salary,
        }


def score_predictions(
    predictions: list[dict], gold: list[dict], *, salary_rel_tolerance: float,
) -> ScoreReport:
    """Aggregate per-field scores. predictions: {offer_id, valid, parsed}; gold: {offer_id, ...}."""
    gold_by_id = {g["offer_id"]: g for g in gold}
    metrics = {f: SetMetric() for f in SET_FIELDS}
    salary_acc = {"currency": 0, "kind": 0, "amount": 0}
    valid_count = title_hits = matched = 0

    for pred in predictions:
        g = gold_by_id.get(pred["offer_id"])
        if g is None:
            continue
        matched += 1
        valid_count += int(pred.get("valid", False))
        parsed = pred.get("parsed") or {}
        for f in SET_FIELDS:
            metrics[f].add(parsed.get(f), g.get(f))
        sal = _score_salary(parsed.get("salary"), g.get("salary"), salary_rel_tolerance)
        for k in salary_acc:
            salary_acc[k] += sal[k]
        pt = (parsed.get("title") or "").strip().lower()
        gt = (g.get("title") or "").strip().lower()
        title_hits += int(bool(gt) and pt == gt)

    denom = matched or 1
    return ScoreReport(
        n=matched,
        json_validity=valid_count / denom,
        title_exact=title_hits / denom,
        fields={f: m.as_dict() for f, m in metrics.items()},
        salary={k: round(v / denom, 4) for k, v in salary_acc.items()},
    )

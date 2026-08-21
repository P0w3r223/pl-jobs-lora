"""Uncertainty on a 142-record test set (ADR-0003): bootstrap CIs and paired variant differences.

Pure, offline and deterministic — resampling is seeded, so the same predictions always yield the
same interval. No model, no network.

Why this exists. The comparison table reports point estimates over **142 records**. A reader
cannot tell from ``0.39`` versus ``0.23`` whether the gap is real or whether a different sample of
142 postings would have reversed it, and a project whose argument is *honest measurement* should
not make them guess. Two things are computed:

- **Per-variant intervals.** Records are resampled with replacement and the variant re-scored on
  each resample; the percentile interval over those scores is the reported CI.
- **Paired differences.** The decisive question is comparative, so every variant is scored on the
  **same** drawn records within a resample. Pairing cancels the "was this a hard draw of postings"
  component the variants share, which an interval-overlap eyeball cannot do: two per-variant CIs
  can overlap while the paired difference is consistently one-signed.

Resampling is by **record**, never by prediction row: the unit of independence here is a posting.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from pl_jobs_lora.eval import scoring

METRICS = ("mean_field_f1", "json_validity")


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


@dataclass
class Interval:
    """A point estimate with a percentile bootstrap interval; ``None`` when unmeasurable."""

    point: float | None
    low: float | None
    high: float | None

    def as_dict(self) -> dict:
        return {"point": _round(self.point), "low": _round(self.low), "high": _round(self.high)}


@dataclass
class PairedDifference:
    """``variant`` minus ``reference`` on one metric, scored on identical resampled records."""

    variant: str
    reference: str
    metric: str
    difference: Interval
    sign_agreement: float   # share of resamples whose difference has the sign of the point estimate
    comparable: bool = True  # False when either side answered only part of the gold set

    @property
    def separated(self) -> bool:
        """Whether the interval excludes zero — the gap survives resampling of the test set.

        Never true for an incomparable pair: if one side answered a subset, the two were scored on
        different records within each resample, so the difference is not the paired quantity this
        report claims to publish and must not be read as a finding.
        """
        if not self.comparable:
            return False
        lo, hi = self.difference.low, self.difference.high
        return lo is not None and hi is not None and (lo > 0 or hi < 0)

    def as_dict(self) -> dict:
        return {
            "variant": self.variant, "reference": self.reference, "metric": self.metric,
            "difference": self.difference.as_dict(),
            "sign_agreement": round(self.sign_agreement, 4),
            "comparable": self.comparable,
            "separated": self.separated,
        }


@dataclass
class BootstrapReport:
    n_gold: int
    resamples: int
    seed: int
    ci: float
    intervals: dict = field(default_factory=dict)     # variant -> {metric -> Interval}
    paired: list = field(default_factory=list)        # [PairedDifference]
    reference: str | None = None
    coverage: dict = field(default_factory=dict)      # variant -> share of gold records answered

    def as_dict(self) -> dict:
        return {
            "n_gold": self.n_gold, "resamples": self.resamples, "seed": self.seed,
            "ci": self.ci, "reference": self.reference,
            "coverage": {k: round(v, 4) for k, v in self.coverage.items()},
            "intervals": {
                v: {m: i.as_dict() for m, i in metrics.items()}
                for v, metrics in self.intervals.items()
            },
            "paired": [p.as_dict() for p in self.paired],
        }


def _percentile(ordered: list[float], q: float) -> float:
    """Linear-interpolated percentile over an already-sorted list (q in [0, 100])."""
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _interval(point: float | None, draws: list[float], ci: float) -> Interval:
    if point is None or not draws:
        return Interval(point=point, low=None, high=None)
    tail = (100.0 - ci) / 2.0
    ordered = sorted(draws)
    return Interval(point, _percentile(ordered, tail), _percentile(ordered, 100.0 - tail))


def _metrics(
    preds_by_id: dict[str, list[dict]], sample: list[dict], tol: float,
) -> dict[str, float | None]:
    """The two headline metrics for one variant over one (possibly resampled) record list.

    Predictions are looked up per drawn record, so a record drawn twice contributes twice — which
    is what resampling with replacement means. An id carrying several prediction rows contributes
    all of them, exactly as the main table's scorer does, so the point estimate here and the point
    estimate there cannot disagree on a file that answers a record twice.
    """
    predictions = [row for g in sample for row in preds_by_id.get(g["offer_id"], ())]
    scores = scoring.score_predictions(predictions, sample, salary_rel_tolerance=tol).as_dict()
    return {
        "mean_field_f1": scoring.mean_measured_f1(scores["fields"]),
        "json_validity": scores["json_validity"],
    }


def bootstrap_variants(
    variant_preds: dict[str, list[dict]], gold: list[dict], *,
    salary_rel_tolerance: float, resamples: int, seed: int, ci: float,
    reference: str | None = None,
) -> BootstrapReport:
    """Per-variant CIs plus paired differences against ``reference`` (default: best mean field F1).

    One resampling loop serves both: because every variant is scored on the same drawn records
    within an iteration, the per-resample difference between two variants is already paired, and
    reading it off the stored draws is exact — no second, differently-seeded pass.
    """
    if resamples < 1:
        raise ValueError("resamples must be >= 1")
    if not 0 < ci < 100:
        raise ValueError("ci must be a percentage in (0, 100)")
    if not gold:
        raise ValueError("cannot bootstrap an empty gold set")

    gold_ids = {g["offer_id"] for g in gold}
    by_variant: dict[str, dict[str, list[dict]]] = {}
    for name, preds in variant_preds.items():
        rows: dict[str, list[dict]] = {}
        for p in preds:
            if p["offer_id"] in gold_ids:
                rows.setdefault(p["offer_id"], []).append(p)
        by_variant[name] = rows
    # A variant that answered only part of the gold set is scored on its own subset within every
    # resample, so its metric is not on the same footing as a complete variant's. Recorded here so
    # the paired section can refuse to claim a difference it cannot honestly make.
    coverage = {name: len(rows) / len(gold) for name, rows in by_variant.items()}
    point = {
        name: _metrics(rows, gold, salary_rel_tolerance) for name, rows in by_variant.items()
    }

    # draws[variant][metric][i] is that variant's metric on resample i — None when unmeasurable
    # there. Keeping the None in place (rather than dropping it) is what keeps index i meaning the
    # same resample across variants, which the pairing below depends on.
    rng = random.Random(seed)
    draws: dict[str, dict[str, list[float | None]]] = {
        name: {m: [] for m in METRICS} for name in by_variant
    }
    n = len(gold)
    for _ in range(resamples):
        # Indices are derived from `random()`, whose stream is the documented-stable part of the
        # Mersenne Twister API; `randrange` makes no such cross-version promise. These intervals
        # are a committed artifact, so a Python upgrade must not silently move published numbers.
        sample = [gold[int(rng.random() * n)] for _ in range(n)]
        for name, rows in by_variant.items():
            values = _metrics(rows, sample, salary_rel_tolerance)
            for metric in METRICS:
                draws[name][metric].append(values[metric])

    intervals = {
        name: {
            m: _interval(point[name][m], [d for d in draws[name][m] if d is not None], ci)
            for m in METRICS
        }
        for name in by_variant
    }

    if reference is None:
        measurable = [n for n in by_variant if point[n]["mean_field_f1"] is not None]
        reference = max(measurable, key=lambda n: point[n]["mean_field_f1"], default=None)

    paired = (
        _paired_differences(by_variant, point, draws, reference, ci, coverage)
        if reference is not None else []
    )
    return BootstrapReport(
        n_gold=len(gold), resamples=resamples, seed=seed, ci=ci,
        intervals=intervals, paired=paired, reference=reference, coverage=coverage,
    )


def _sign_agreement(observed: float, diffs: list[float]) -> float:
    """Share of resamples that land on the same side of zero as the observed difference.

    An exactly zero observed difference has no side to agree with, so the question becomes how
    often the two variants also tie — otherwise a dead heat would report perfect agreement on a
    direction it does not have.
    """
    if not diffs:
        return 0.0
    if observed == 0:
        return sum(1 for d in diffs if d == 0) / len(diffs)
    sign = observed > 0
    return sum(1 for d in diffs if d != 0 and (d > 0) == sign) / len(diffs)


def _paired_differences(
    by_variant: dict, point: dict, draws: dict, reference: str, ci: float, coverage: dict,
) -> list[PairedDifference]:
    out: list[PairedDifference] = []
    for name in by_variant:
        if name == reference:
            continue
        # Pairing only cancels the draw when both sides were scored on the *same* records. A
        # partial variant is scored on its own subset of each draw, so the difference is between
        # two different populations and the claim this section makes would not hold for it.
        comparable = coverage.get(name, 0.0) >= 1.0 and coverage.get(reference, 0.0) >= 1.0
        for metric in METRICS:
            observed_a, observed_b = point[name][metric], point[reference][metric]
            diffs = [
                a - b
                for a, b in zip(draws[name][metric], draws[reference][metric], strict=True)
                if a is not None and b is not None
            ]
            if observed_a is None or observed_b is None:
                out.append(PairedDifference(
                    variant=name, reference=reference, metric=metric,
                    difference=Interval(None, None, None), sign_agreement=0.0,
                    comparable=comparable,
                ))
                continue
            observed = observed_a - observed_b
            out.append(PairedDifference(
                variant=name, reference=reference, metric=metric,
                difference=_interval(observed, diffs, ci),
                sign_agreement=_sign_agreement(observed, diffs),
                comparable=comparable,
            ))
    return out


def render_markdown(report: BootstrapReport) -> str:
    """The uncertainty section appended to the comparison report."""
    fmt = scoring.fmt_metric
    ci_label = f"{report.ci:.0f} % CI"
    lines = [
        "## Uncertainty (bootstrap)\n",
        f"{report.resamples} resamples of the {report.n_gold} test records, drawn with "
        f"replacement (seed {report.seed}); intervals are {report.ci:.0f} % percentile bands. "
        "The point estimates are the numbers in the table above — the interval says how far a "
        "different sample of postings could have moved them.\n",
        f"| variant | field F1 | {ci_label} | JSON valid | {ci_label} |",
        "|---|---|---|---|---|",
    ]
    for name, metrics in report.intervals.items():
        f1, valid = metrics["mean_field_f1"], metrics["json_validity"]
        lines.append(
            f"| {name} | {fmt(f1.point)} | [{fmt(f1.low)}, {fmt(f1.high)}] | "
            f"{fmt(valid.point)} | [{fmt(valid.low)}, {fmt(valid.high)}] |"
        )

    if report.paired:
        lines += [
            "",
            f"### Paired differences vs `{report.reference}`\n",
            "Where both variants answered the whole gold set, each resample scores them on the "
            "**same** drawn records, so the shared difficulty of the draw cancels. `separated` "
            "means the interval excludes zero — the gap survives resampling. Two overlapping "
            "per-variant CIs can still yield a separated paired difference; that is the point of "
            "pairing. `sign agreement` is the share of resamples landing on the observed side of "
            "zero — or, for an exactly tied difference, the share that also tie.\n",
            f"| variant | metric | difference | {ci_label} | sign agreement | separated |",
            "|---|---|---|---|---|---|",
        ]
        for p in report.paired:
            d = p.difference
            verdict = "n/a" if not p.comparable else ("yes" if p.separated else "no")
            lines.append(
                f"| {p.variant} | {p.metric} | {fmt(d.point)} | [{fmt(d.low)}, {fmt(d.high)}] | "
                f"{p.sign_agreement:.2f} | {verdict} |"
            )
        incomparable = sorted({p.variant for p in report.paired if not p.comparable})
        if incomparable:
            lines += [
                "",
                "`n/a` marks a pair that is **not** paired: "
                + ", ".join(f"`{v}`" for v in incomparable)
                + " answered only part of the gold set, so within each resample the two sides "
                "were scored on different records. That difference is between two populations "
                "rather than one, and is not reported as a finding.",
            ]
    return "\n".join(lines)

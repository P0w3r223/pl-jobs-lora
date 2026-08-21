"""Comparison report over variant predictions (S4; ADR-0003): accuracy x cost x latency.

Pure and offline. For every ``predictions/{variant}.jsonl`` it scores the variant with the
ADR-0003 scorer, then folds in API cost (tokens x list price) and p50/p95 latency. Variants
whose rows carry no token counts — the local base/LoRA/GGUF runs, ~$0 marginal — report no cost,
so the same report merges the API baselines with predictions dropped in later from Colab. Writes
the numbers-only ``results/eval/report.{json,md}``; the headline question it answers is whether a
1.5B local LoRA rivals a frontier API at a fraction of the cost/latency.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from pl_jobs_lora.eval import bootstrap as bootstrap_mod
from pl_jobs_lora.eval import ceiling as ceiling_mod
from pl_jobs_lora.eval import pricing, scoring
from pl_jobs_lora.eval.scoring import fmt_metric

_ROOT = Path(__file__).resolve().parents[3]
_PROCESSED = _ROOT / "data" / "processed"
_RESULTS = _ROOT / "results" / "eval"
_PRED_DIR = _RESULTS / "predictions"


@dataclass
class VariantReport:
    variant: str
    scores: dict                        # ADR-0003 ScoreReport.as_dict()
    mean_field_f1: float | None
    usd_per_1k_postings: float | None    # None for local variants (no token counts)
    latency_s: dict = field(default_factory=dict)   # {"p50": .., "p95": ..} or {} if unmeasured
    failures: dict = field(default_factory=dict)    # failure class -> count (invalid rows only)
    at_token_cap: int = 0               # decode failures whose output hit the cap (truncation)

    @property
    def coverage(self) -> float:
        """Fraction of the gold set this variant actually predicted — a partial run must show."""
        return self.scores["coverage"]

    @property
    def n_duplicate_rows(self) -> int:
        """Rows answering an already-answered id — they double-weight a record at any coverage."""
        return self.scores["n_duplicate_rows"]

    @property
    def trustworthy(self) -> bool:
        """Whether this row's metrics are denominated by the whole gold set exactly once."""
        return self.coverage >= 1.0 and self.n_duplicate_rows == 0

    def as_dict(self) -> dict:
        return {
            "variant": self.variant, "scores": self.scores,
            "mean_field_f1": None if self.mean_field_f1 is None else round(self.mean_field_f1, 4),
            "usd_per_1k_postings": self.usd_per_1k_postings, "latency_s": self.latency_s,
            "failures": self.failures, "at_token_cap": self.at_token_cap,
        }


@dataclass
class ComparisonReport:
    metadata: dict
    variants: list[VariantReport]
    bootstrap: bootstrap_mod.BootstrapReport | None = None
    ceilings: dict | None = None       # field -> ceiling.FieldCeiling

    def as_dict(self) -> dict:
        return {
            "metadata": self.metadata,
            "variants": [v.as_dict() for v in self.variants],
            "bootstrap": self.bootstrap.as_dict() if self.bootstrap else None,
            "ceilings": (
                {f: c.as_dict() for f, c in self.ceilings.items()} if self.ceilings else None
            ),
        }

    def best_recall(self, field: str) -> float | None:
        """Highest recall any variant reached on a field — what to hold against its ceiling."""
        values = [
            v.scores["fields"][field]["recall"] for v in self.variants
            if field in v.scores["fields"] and v.scores["fields"][field]["recall"] is not None
        ]
        return max(values) if values else None

    def render_markdown(self) -> str:
        m = self.metadata
        head = (
            "| variant | coverage | JSON valid | seniority F1 | tech F1 | work-mode F1 | "
            "salary detect | salary cur/kind/amt | field F1 | $/1k | p50 s | p95 s |"
        )
        sep = "|---|---|---|---|---|---|---|---|---|---|---|---|"
        lines = [
            f"# Evaluation report (gold n={m['n_gold']})\n",
            "Every variant scored by the same pure scorer on the same frozen test set "
            f"(ADR-0003). API cost priced at {m['api_pricing']}.\n",
            "`coverage` is the fraction of the gold set the variant actually predicted — a run "
            "that died part-way scores only its own subset, and must not read as a full one. "
            "Salary is split: `detect` is the present/absent decision over all records, "
            "`cur/kind/amt` are accuracies over the records whose gold *has* a salary "
            f"(support n={m['salary_support']}), so predicting nothing earns nothing. "
            "`-` means unmeasurable on this test set, never a perfect score.\n",
            head, sep,
        ]
        for v in self.variants:
            s, f = v.scores, v.scores["fields"]
            sal = s["salary"]
            cells = [
                v.variant,
                f"{v.coverage:.2f}" + ("" if v.trustworthy else " ⚠"),
                f"{s['json_validity']:.2f}",
                fmt_metric(f["seniority"]["f1"]), fmt_metric(f["tech_expected"]["f1"]),
                fmt_metric(f["work_mode"]["f1"]), fmt_metric(sal["detection"]),
                f"{fmt_metric(sal['currency'])}/{fmt_metric(sal['kind'])}"
                f"/{fmt_metric(sal['amount'])}",
                fmt_metric(v.mean_field_f1),
                "-" if v.usd_per_1k_postings is None else f"{v.usd_per_1k_postings:.2f}",
                fmt_metric(v.latency_s.get("p50")), fmt_metric(v.latency_s.get("p95")),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        sections = [*lines, "", self._render_failures()]
        if self.ceilings:
            recalls = {f: self.best_recall(f) for f in self.ceilings}
            sections += ["", ceiling_mod.render_markdown(self.ceilings, recalls)]
        if self.bootstrap is not None:
            sections += ["", bootstrap_mod.render_markdown(self.bootstrap)]
        return "\n".join(sections) + "\n"

    def _render_failures(self) -> str:
        """Why the invalid rows are invalid — `JSON valid` says how often, never because of what."""
        classes = [c for c in scoring.PARSE_FAILURES
                   if any(v.failures.get(c) for v in self.variants)]
        if not classes:
            return "## Failure taxonomy\n\nEvery variant parsed cleanly on every record.\n"

        head = "| variant | invalid | " + " | ".join(classes) + " | at token cap |"
        lines = [
            "## Failure taxonomy\n",
            "`JSON valid` above says *how often* a variant failed; this says *how*. "
            "`at token cap` counts the `json_decode_error` rows whose output reached the decoding "
            "cap — plausibly truncated rather than malformed, the one confound the validity rate "
            "cannot separate on its own. `unrecorded` marks rows written before the parser "
            "classified its failures.\n",
            head, "|---|---|" + "---|" * (len(classes) + 1),
        ]
        for v in self.variants:
            counts = [str(v.failures.get(c, 0)) for c in classes]
            cap = str(v.at_token_cap) if v.at_token_cap else "-"
            lines.append(
                f"| {v.variant} | {sum(v.failures.values())} | " + " | ".join(counts)
                + f" | {cap} |"
            )
        return "\n".join(lines)


def tally_failures(
    preds: list[dict], *, decode_max_tokens: int | None, gold_ids: set[str] | None = None,
) -> tuple[dict, int]:
    """Histogram of *why* the invalid rows are invalid, plus how many plausibly hit the token cap.

    Counts only rows the scorer also counted — a prediction for an offer outside the gold set is
    invisible to every other metric, so counting it here would let this table list failures the
    main table says do not exist.

    A row written before the taxonomy existed carries no ``failure`` key, and a row written by an
    older vocabulary may carry a class this build no longer knows; both are ``unrecorded`` —
    "invalid, reason unrecoverable" — rather than dropped, because dropping is what would let an
    old predictions file read as a clean one. ``at_token_cap`` counts decode failures whose output
    reached the decoding cap: the truncation confound, otherwise indistinguishable from malformed.
    """
    counts: dict[str, int] = {}
    at_cap = 0
    for p in preds:
        if gold_ids is not None and p.get("offer_id") not in gold_ids:
            continue
        if p.get("valid", False):
            continue
        cls = p.get("failure") or scoring.UNRECORDED
        if cls not in scoring.PARSE_FAILURES:
            cls = scoring.UNRECORDED
        counts[cls] = counts.get(cls, 0) + 1
        out_tokens = p.get("output_tokens")
        if (
            cls == scoring.JSON_DECODE_ERROR
            and decode_max_tokens
            and out_tokens is not None
            and out_tokens >= decode_max_tokens
        ):
            at_cap += 1
    return {k: counts[k] for k in scoring.PARSE_FAILURES if k in counts}, at_cap


def _read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_gold(processed_dir: Path = _PROCESSED) -> list[dict]:
    """Gold labels for scoring: flatten each frozen test record to {offer_id, ...fields}."""
    rows = _read_jsonl(processed_dir / "test.jsonl")
    return [{"offer_id": r["offer_id"], **r["gold"]} for r in rows]


def load_test_records(processed_dir: Path = _PROCESSED) -> list[dict]:
    """Frozen test records *with* their prose — the data ceiling needs the model's actual input."""
    return _read_jsonl(processed_dir / "test.jsonl")


def discover_predictions(pred_dir: Path = _PRED_DIR) -> list[Path]:
    return sorted(pred_dir.glob("*.jsonl"))


def score_variant(
    variant: str, preds: list[dict], gold: list[dict], *,
    salary_rel_tolerance: float, input_usd_per_mtok: float, output_usd_per_mtok: float,
    latency_percentiles: tuple[int, ...], decode_max_tokens: int | None = None,
) -> VariantReport:
    """Fold ADR-0003 scores + cost + latency into one variant summary (pure)."""
    scores = scoring.score_predictions(
        preds, gold, salary_rel_tolerance=salary_rel_tolerance
    ).as_dict()
    mean_field_f1 = scoring.mean_measured_f1(scores["fields"])

    priced = [p for p in preds if p.get("input_tokens") is not None]
    usd_per_1k = None
    if priced:
        per_call = [
            pricing.cost_usd(
                p["input_tokens"], p.get("output_tokens", 0),
                input_usd_per_mtok=input_usd_per_mtok, output_usd_per_mtok=output_usd_per_mtok,
            )
            for p in priced
        ]
        usd_per_1k = round(statistics.mean(per_call) * 1000, 4)

    latencies = [p["latency_s"] for p in preds if "latency_s" in p]
    latency_s = {}
    for p in latency_percentiles:
        val = pricing.percentile(latencies, p)
        if val is not None:
            latency_s[f"p{p}"] = round(val, 3)

    failures, at_token_cap = tally_failures(
        preds, decode_max_tokens=decode_max_tokens, gold_ids={g["offer_id"] for g in gold},
    )

    return VariantReport(
        variant=variant, scores=scores, mean_field_f1=mean_field_f1,
        usd_per_1k_postings=usd_per_1k, latency_s=latency_s,
        failures=failures, at_token_cap=at_token_cap,
    )


def build_report(
    prediction_files: list[Path], gold: list[dict], *,
    salary_rel_tolerance: float, input_usd_per_mtok: float, output_usd_per_mtok: float,
    latency_percentiles: tuple[int, ...], api_pricing: str = "",
    decode_max_tokens: int | None = None,
    bootstrap_resamples: int = 0, bootstrap_seed: int = 0, bootstrap_ci: float = 95.0,
    ceilings: dict | None = None,
) -> ComparisonReport:
    preds_by_variant = {path.stem: _read_jsonl(path) for path in prediction_files}
    variants = [
        score_variant(
            name, preds, gold,
            salary_rel_tolerance=salary_rel_tolerance,
            input_usd_per_mtok=input_usd_per_mtok, output_usd_per_mtok=output_usd_per_mtok,
            latency_percentiles=latency_percentiles, decode_max_tokens=decode_max_tokens,
        )
        for name, preds in preds_by_variant.items()
    ]
    metadata = {
        "n_gold": len(gold),
        "n_variants": len(variants),
        "salary_support": sum(1 for g in gold if g.get("salary") is not None),
        "salary_rel_tolerance": salary_rel_tolerance,
        "api_pricing": api_pricing,
        "incomplete_variants": [v.variant for v in variants if v.coverage < 1.0],
        "duplicated_row_variants": {
            v.variant: v.n_duplicate_rows for v in variants if v.n_duplicate_rows
        },
        "note": "API variants priced by tokens x list price; local (base/LoRA) variants ~$0.",
    }
    resampled = None
    if bootstrap_resamples and gold:
        resampled = bootstrap_mod.bootstrap_variants(
            preds_by_variant, gold, salary_rel_tolerance=salary_rel_tolerance,
            resamples=bootstrap_resamples, seed=bootstrap_seed, ci=bootstrap_ci,
        )
    return ComparisonReport(
        metadata=metadata, variants=variants, bootstrap=resampled, ceilings=ceilings,
    )


def write_report(report: ComparisonReport, out_dir: Path = _RESULTS) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(report.render_markdown(), encoding="utf-8")

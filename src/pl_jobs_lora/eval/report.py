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

from pl_jobs_lora.eval import pricing, scoring

_ROOT = Path(__file__).resolve().parents[3]
_PROCESSED = _ROOT / "data" / "processed"
_RESULTS = _ROOT / "results" / "eval"
_PRED_DIR = _RESULTS / "predictions"


@dataclass
class VariantReport:
    variant: str
    scores: dict                        # ADR-0003 ScoreReport.as_dict()
    mean_field_f1: float
    usd_per_1k_postings: float | None    # None for local variants (no token counts)
    latency_s: dict = field(default_factory=dict)   # {"p50": .., "p95": ..} or {} if unmeasured

    def as_dict(self) -> dict:
        return {
            "variant": self.variant, "scores": self.scores,
            "mean_field_f1": round(self.mean_field_f1, 4),
            "usd_per_1k_postings": self.usd_per_1k_postings, "latency_s": self.latency_s,
        }


@dataclass
class ComparisonReport:
    metadata: dict
    variants: list[VariantReport]

    def as_dict(self) -> dict:
        return {"metadata": self.metadata, "variants": [v.as_dict() for v in self.variants]}

    def render_markdown(self) -> str:
        m = self.metadata
        head = (
            "| variant | JSON valid | seniority F1 | tech F1 | work-mode F1 | "
            "salary(cur/kind/amt) | field F1 | $/1k | p50 s | p95 s |"
        )
        sep = "|---|---|---|---|---|---|---|---|---|---|"
        lines = [
            f"# Evaluation report (n={m['n_gold']})\n",
            "Every variant scored by the same pure scorer on the same frozen test set "
            f"(ADR-0003). API cost priced at {m['api_pricing']}.\n",
            head, sep,
        ]
        for v in self.variants:
            s, f = v.scores, v.scores["fields"]
            sal = s["salary"]
            cost = "-" if v.usd_per_1k_postings is None else f"{v.usd_per_1k_postings:.2f}"
            p50 = v.latency_s.get("p50")
            p95 = v.latency_s.get("p95")
            lines.append(
                f"| {v.variant} | {s['json_validity']:.2f} | {f['seniority']['f1']:.2f} | "
                f"{f['tech_expected']['f1']:.2f} | {f['work_mode']['f1']:.2f} | "
                f"{sal['currency']:.2f}/{sal['kind']:.2f}/{sal['amount']:.2f} | "
                f"{v.mean_field_f1:.2f} | {cost} | "
                f"{'-' if p50 is None else f'{p50:.2f}'} | {'-' if p95 is None else f'{p95:.2f}'} |"
            )
        return "\n".join(lines) + "\n"


def _read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_gold(processed_dir: Path = _PROCESSED) -> list[dict]:
    """Gold labels for scoring: flatten each frozen test record to {offer_id, ...fields}."""
    rows = _read_jsonl(processed_dir / "test.jsonl")
    return [{"offer_id": r["offer_id"], **r["gold"]} for r in rows]


def discover_predictions(pred_dir: Path = _PRED_DIR) -> list[Path]:
    return sorted(pred_dir.glob("*.jsonl"))


def score_variant(
    variant: str, preds: list[dict], gold: list[dict], *,
    salary_rel_tolerance: float, input_usd_per_mtok: float, output_usd_per_mtok: float,
    latency_percentiles: tuple[int, ...],
) -> VariantReport:
    """Fold ADR-0003 scores + cost + latency into one variant summary (pure)."""
    scores = scoring.score_predictions(
        preds, gold, salary_rel_tolerance=salary_rel_tolerance
    ).as_dict()
    mean_field_f1 = statistics.mean(f["f1"] for f in scores["fields"].values())

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

    return VariantReport(
        variant=variant, scores=scores, mean_field_f1=mean_field_f1,
        usd_per_1k_postings=usd_per_1k, latency_s=latency_s,
    )


def build_report(
    prediction_files: list[Path], gold: list[dict], *,
    salary_rel_tolerance: float, input_usd_per_mtok: float, output_usd_per_mtok: float,
    latency_percentiles: tuple[int, ...], api_pricing: str = "",
) -> ComparisonReport:
    variants = [
        score_variant(
            path.stem, _read_jsonl(path), gold,
            salary_rel_tolerance=salary_rel_tolerance,
            input_usd_per_mtok=input_usd_per_mtok, output_usd_per_mtok=output_usd_per_mtok,
            latency_percentiles=latency_percentiles,
        )
        for path in prediction_files
    ]
    metadata = {
        "n_gold": len(gold),
        "n_variants": len(variants),
        "salary_rel_tolerance": salary_rel_tolerance,
        "api_pricing": api_pricing,
        "note": "API variants priced by tokens x list price; local (base/LoRA) variants ~$0.",
    }
    return ComparisonReport(metadata=metadata, variants=variants)


def write_report(report: ComparisonReport, out_dir: Path = _RESULTS) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(report.render_markdown(), encoding="utf-8")

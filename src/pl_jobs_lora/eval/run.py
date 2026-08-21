"""S4 evaluation CLI (ADR-0003): run the API baselines, then build the comparison report.

    python -m pl_jobs_lora.eval.run --baselines   # paid + network: zero-/few-shot over test set
    python -m pl_jobs_lora.eval.run --report      # pure + offline: score every predictions file

``--baselines`` needs ANTHROPIC_API_KEY and the ``api`` extra; ``--report`` is fully offline and
aggregates whatever ``results/eval/predictions/*.jsonl`` exist (API baselines now, base/LoRA runs
dropped in from the hosted GPU later).
"""

from __future__ import annotations

import argparse

from pl_jobs_lora.config import load_config
from pl_jobs_lora.eval import baselines, ceiling, report
from pl_jobs_lora.normalize import load_tech_aliases


def main() -> None:
    ap = argparse.ArgumentParser(description="S4 evaluation: API baselines + comparison report.")
    ap.add_argument("--baselines", action="store_true", help="run the zero-/few-shot API baselines")
    ap.add_argument("--report", action="store_true", help="build the comparison report (offline)")
    ap.add_argument("--limit", type=int, default=0, help="cap test examples for a smoke run")
    ap.add_argument(
        "--no-bootstrap", action="store_true",
        help="skip the uncertainty section (the resampling dominates --report's runtime)",
    )
    args = ap.parse_args()

    cfg = load_config()
    if args.baselines:
        out = baselines.run_baselines(cfg, limit=args.limit)
        for variant, preds in out.items():
            valid = sum(int(p["valid"]) for p in preds)
            print(f"[eval] {variant}: {len(preds)} predictions, {valid} valid JSON")
    if args.report:
        gold = report.load_gold()
        files = report.discover_predictions()
        # Model-free: how much of each open-vocabulary label is in the prose at all. Needs the
        # frozen records (not the flattened gold) because it reads the model's actual input.
        ceilings = ceiling.answerable_ceilings(
            report.load_test_records(), load_tech_aliases(),
        )
        rep = report.build_report(
            files, gold,
            salary_rel_tolerance=cfg.scoring.salary_rel_tolerance,
            input_usd_per_mtok=cfg.eval.input_usd_per_mtok,
            output_usd_per_mtok=cfg.eval.output_usd_per_mtok,
            latency_percentiles=cfg.eval.latency_percentiles,
            decode_max_tokens=cfg.eval.max_tokens,
            bootstrap_resamples=0 if args.no_bootstrap else cfg.scoring.bootstrap_resamples,
            bootstrap_seed=cfg.scoring.bootstrap_seed,
            bootstrap_ci=cfg.scoring.bootstrap_ci,
            ceilings=ceilings,
            api_pricing=(
                f"{cfg.eval.api_model} "
                f"${cfg.eval.input_usd_per_mtok}/${cfg.eval.output_usd_per_mtok} per MTok"
            ),
        )
        report.write_report(rep)
        print(f"[eval] report over {len(files)} variant(s), n_gold={len(gold)}")
        print(rep.render_markdown())


if __name__ == "__main__":
    main()

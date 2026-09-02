"""Base-model selection probe (ADR-0001): run both candidates zero/few-shot over a dev slice,
score per-field, record numbers, pick the winner.

Inference runs on local CPU via GGUF (matched Q8_0, ADR-0001) — llama-cpp-python is imported
lazily so the pure parts (report aggregation, winner pick) stay importable and testable. Each
variant emits ``results/probe/predictions/{candidate}__{mode}.jsonl``; scoring is the pure ADR-0003
scorer. The probe measures the base model's steerability toward our JSON schema — the better
*starting point* for QLoRA, not a final quality claim.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from pl_jobs_lora.config import Config, ModelCandidate, load_config
from pl_jobs_lora.dataset.collect import DevExample, collect_dev_slice
from pl_jobs_lora.eval import scoring
from pl_jobs_lora.eval.prompt import build_messages, parse_result
from pl_jobs_lora.eval.scoring import fmt_metric
from pl_jobs_lora.normalize import load_tech_aliases

_ROOT = Path(__file__).resolve().parents[2]
_CACHE = _ROOT / "data" / "probe_slice.json"          # gitignored: prose-derived slice, replay
_RESULTS = _ROOT / "results" / "probe"
_PRED_DIR = _RESULTS / "predictions"
_MODELS_DIR = _ROOT / "data" / "models"               # gitignored: downloaded GGUFs


def load_slice(path: Path = _CACHE) -> list[DevExample]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [DevExample(**r) for r in rows]


def save_slice(examples: list[DevExample], path: Path = _CACHE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(e) for e in examples], ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _gguf_path(cand: ModelCandidate) -> Path:
    """Download (cached) the candidate's GGUF and return its local path."""
    from huggingface_hub import hf_hub_download

    if not cand.gguf_repo or not cand.gguf_file:
        raise ValueError(f"{cand.key}: gguf_repo/gguf_file unset in config (resolve in the probe).")
    return Path(
        hf_hub_download(cand.gguf_repo, cand.gguf_file, local_dir=_MODELS_DIR / cand.key)
    )


def check_context_fits(llm, prompts: list[list[dict]], cfg: Config) -> int:
    """Raise unless every prompt leaves room for a whole completion. Returns the longest prompt.

    The prompt and the completion share one ``n_ctx``, so raising the decoding cap eats headroom
    the longest posting was already using. Checked once up front rather than per record: a run that
    only discovers this at record 90 has burned an hour to report a config error.

    The count is over the message contents; the chat template wraps them in role markers this
    tokenizer call does not see, so ``context_margin_tokens`` covers what the floor misses.
    """
    if not prompts:
        return 0
    budget = cfg.probe.context_tokens - cfg.probe.max_tokens - cfg.probe.context_margin_tokens
    worst = max(
        len(llm.tokenize("\n".join(m["content"] for m in msgs).encode("utf-8")))
        for msgs in prompts
    )
    if worst > budget:
        raise ValueError(
            f"prompt of {worst} tokens exceeds the {budget} left by n_ctx="
            f"{cfg.probe.context_tokens} minus max_tokens={cfg.probe.max_tokens} and a "
            f"{cfg.probe.context_margin_tokens}-token template margin: lower probe.max_tokens "
            f"or raise probe.context_tokens."
        )
    return worst


def run_inference(
    cand: ModelCandidate, mode: str, eval_set: list[DevExample],
    shots: list[DevExample], cfg: Config,
    on_prediction: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Generate + parse predictions for one (candidate, shot-mode). Requires llama-cpp-python.

    ``on_prediction`` is invoked with each prediction as it is produced (before it is appended
    to the return list) so a long run can be checkpointed to disk record-by-record; when ``None``
    the behaviour is unchanged.
    """
    from llama_cpp import Llama

    n_shots = cfg.probe.few_shot_examples if mode == "few" else 0
    alias_index = load_tech_aliases()
    llm = Llama(
        model_path=str(_gguf_path(cand)), n_ctx=cfg.probe.context_tokens,
        n_threads=None, verbose=False,
    )
    prompts = [build_messages(ex, shots, n_shots=n_shots) for ex in eval_set]
    worst = check_context_fits(llm, prompts, cfg)
    print(f"[probe] worst prompt {worst} tokens, cap {cfg.probe.max_tokens}, "
          f"n_ctx {cfg.probe.context_tokens}")
    preds: list[dict] = []
    for ex, messages in zip(eval_set, prompts, strict=True):
        t0 = time.perf_counter()
        resp = llm.create_chat_completion(
            messages=messages, temperature=cfg.probe.temperature, max_tokens=cfg.probe.max_tokens,
        )
        latency = time.perf_counter() - t0
        raw = resp["choices"][0]["message"]["content"] or ""
        result = parse_result(raw, alias_index)
        pred = {
            "offer_id": ex.offer_id, "valid": result.valid, "parsed": result.parsed,
            "failure": result.failure, "raw": raw,
            "latency_s": round(latency, 3),
            "output_tokens": resp.get("usage", {}).get("completion_tokens", 0),
            # The cap this row decoded under, so the report can tell truncation from malformed
            # output without assuming every variant shared one limit.
            "max_tokens": cfg.probe.max_tokens,
        }
        if on_prediction is not None:
            on_prediction(pred)
        preds.append(pred)
    del llm
    return preds


def write_predictions(cand_key: str, mode: str, preds: list[dict]) -> None:
    _PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = _PRED_DIR / f"{cand_key}__{mode}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")


def read_predictions(cand_key: str, mode: str, pred_dir: Path = _PRED_DIR) -> list[dict] | None:
    """Stored predictions for one variant, or None when that variant was never run."""
    path = pred_dir / f"{cand_key}__{mode}.jsonl"
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def build_variant_report(preds: list[dict], gold: list[dict], cfg: Config) -> dict:
    """Pure: fold the ADR-0003 scores plus latency/token economy into one variant summary."""
    report = scoring.score_predictions(
        preds, gold, salary_rel_tolerance=cfg.scoring.salary_rel_tolerance
    ).as_dict()
    latencies = [p["latency_s"] for p in preds if "latency_s" in p]
    tokens = [p["output_tokens"] for p in preds if p.get("output_tokens")]
    report["latency_p50_s"] = round(statistics.median(latencies), 3) if latencies else None
    report["mean_output_tokens"] = round(statistics.mean(tokens), 1) if tokens else None
    mean_f1 = scoring.mean_measured_f1(report["fields"])
    report["mean_field_f1"] = None if mean_f1 is None else round(mean_f1, 4)
    return report


def pick_winner(reports: dict[str, dict]) -> str:
    """Rank by (mean field F1 + JSON validity), tie-break lower p50 latency."""
    def key(item):
        _, r = item
        return (
            (r["mean_field_f1"] or 0.0) + r["json_validity"],
            -(r["latency_p50_s"] or 0),
        )

    return max(reports.items(), key=key)[0]


def render_table(reports: dict[str, dict]) -> str:
    head = (
        "| variant | field F1 | JSON valid | title | salary detect | "
        "salary cur/kind/amt | p50 s | out tok |"
    )
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [head, sep]
    for name, r in reports.items():
        s = r["salary"]
        lines.append(
            f"| {name} | {fmt_metric(r['mean_field_f1'], 3)} | {r['json_validity']:.2f} | "
            f"{fmt_metric(r['title_exact'])} | {fmt_metric(s['detection'])} | "
            f"{fmt_metric(s['currency'])}/{fmt_metric(s['kind'])}/{fmt_metric(s['amount'])} | "
            f"{fmt_metric(r['latency_p50_s'], 1)} | {fmt_metric(r['mean_output_tokens'], 0)} |"
        )
    return "\n".join(lines)


def _eval_split(
    cfg: Config, examples: list[DevExample], limit: int = 0,
) -> tuple[list[DevExample], list[DevExample], list[dict]]:
    """(few-shot exemplars, eval examples, gold) — the one place the split is defined."""
    shots = examples[: cfg.probe.few_shot_examples]
    eval_set = examples[cfg.probe.few_shot_examples :]
    if limit:
        eval_set = eval_set[:limit]
    return shots, eval_set, [{"offer_id": e.offer_id, **e.gold} for e in eval_set]


def write_probe_report(reports: dict[str, dict], n_eval: int, out_dir: Path = _RESULTS) -> dict:
    winner = pick_winner(reports)
    out = {"n_eval": n_eval, "winner": winner, "variants": reports}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(
        f"# Base-model probe\n\nWinner: **{winner}** (n={n_eval})\n\n"
        + render_table(reports) + "\n",
        encoding="utf-8",
    )
    return out


def run_probe(
    cfg: Config, examples: list[DevExample], candidate_keys: list[str] | None, limit: int = 0,
) -> dict:
    shots, eval_set, gold = _eval_split(cfg, examples, limit)
    candidates = [c for c in cfg.models if not candidate_keys or c.key in candidate_keys]

    reports: dict[str, dict] = {}
    for cand in candidates:
        for mode in cfg.probe.shot_modes:
            preds = run_inference(cand, mode, eval_set, shots, cfg)
            write_predictions(cand.key, mode, preds)
            reports[f"{cand.key}/{mode}"] = build_variant_report(preds, gold, cfg)

    return write_probe_report(reports, len(eval_set))


def rescore_probe(
    cfg: Config, examples: list[DevExample], candidate_keys: list[str] | None = None,
) -> dict:
    """Re-score the *stored* predictions with the current scorer — loads no model, hits no network.

    The mirror of ``eval.run --report``: when the scorer changes, the published probe numbers must
    be reproducible from the cached slice without re-paying for inference. Variants whose
    prediction file is missing are skipped; a file that covers only part of the eval set shows up
    as ``coverage < 1`` rather than being silently scored as a complete run.
    """
    _, eval_set, gold = _eval_split(cfg, examples)
    candidates = [c for c in cfg.models if not candidate_keys or c.key in candidate_keys]

    reports: dict[str, dict] = {}
    for cand in candidates:
        for mode in cfg.probe.shot_modes:
            preds = read_predictions(cand.key, mode)
            if preds is None:
                print(f"[probe] no stored predictions for {cand.key}/{mode} — skipped")
                continue
            reports[f"{cand.key}/{mode}"] = build_variant_report(preds, gold, cfg)

    if not reports:
        raise SystemExit("[probe] no stored predictions found — run the probe first")
    return write_probe_report(reports, len(eval_set))


def main() -> None:
    ap = argparse.ArgumentParser(description="Base-model selection probe (ADR-0001).")
    ap.add_argument("--collect", action="store_true", help="fetch fresh slice (else replay cache)")
    ap.add_argument("--candidates", nargs="*", help="subset of candidate keys to probe")
    ap.add_argument("--limit", type=int, default=0, help="cap eval examples (smoke run; 0=all)")
    ap.add_argument(
        "--rescore", action="store_true",
        help="re-score stored predictions with the current scorer (offline; loads no model)",
    )
    args = ap.parse_args()

    if args.rescore and args.collect:
        raise SystemExit("[probe] --rescore replays stored predictions; --collect makes no sense")

    cfg = load_config()
    if args.collect:
        examples = collect_dev_slice(cfg, load_tech_aliases())
        save_slice(examples)
    else:
        examples = load_slice()
    print(f"[probe] {len(examples)} examples")
    if args.rescore:
        out = rescore_probe(cfg, examples, args.candidates)
    else:
        out = run_probe(cfg, examples, args.candidates, args.limit)
    print(f"[probe] winner: {out['winner']}")
    print(render_table(out["variants"]))


if __name__ == "__main__":
    main()

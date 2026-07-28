"""Zero-/few-shot API baselines (S4; ADR-0003): run the frozen test set through a cheap
frontier API, parse plain text, and emit predictions for the pure scorer.

Fairness (ADR-0003): the baselines reuse the *same* prompt as the probe (``build_messages`` +
``parse_output``) and generate **plain text** — no forced tool call — so JSON validity stays a
first-class metric the model can fail. The only variable across variants is the model.

Orchestration + I/O only. The Anthropic client is imported lazily (optional ``api`` extra, the
same seam as the S3 arbiter), so the core install, tests, and CI stay fully offline; pass a
``generate_fn`` to exercise the assembly without a network. Each variant emits
``results/eval/predictions/{model}__{mode}.jsonl`` with per-call token counts + latency; the
frozen test set stays local (gitignored), so only the numbers-only report is ever committed.

    python -m pl_jobs_lora.eval.run --baselines   # needs ANTHROPIC_API_KEY (paid, network)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from pl_jobs_lora.config import Config, load_config
from pl_jobs_lora.dataset.collect import DevExample
from pl_jobs_lora.eval.prompt import build_messages, parse_output
from pl_jobs_lora.normalize import load_tech_aliases

_ROOT = Path(__file__).resolve().parents[3]
_PROCESSED = _ROOT / "data" / "processed"              # frozen S2 test set (gitignored)
_PRED_DIR = _ROOT / "results" / "eval" / "predictions"  # per-run predictions (gitignored)


def _read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def to_dev_examples(records: list[dict]) -> list[DevExample]:
    return [DevExample(**r) for r in records]


def _build_client():
    """Lazy Anthropic client (optional ``api`` extra); the key comes from ANTHROPIC_API_KEY."""
    import anthropic

    return anthropic.Anthropic()


def _api_generate(client, cfg: Config, messages: list[dict]) -> tuple[str, int, int]:
    """One baseline call: plain-text extraction; returns (raw text, input tokens, output tokens).

    ``build_messages`` yields ``[system, user, assistant, ...shots..., user]``; the system block
    goes to ``system=`` and the rest (already user/assistant-alternating) to ``messages=``.
    Greedy (temperature 0) so decoding matches the probe — only the model differs (ADR-0003).
    """
    resp = client.messages.create(
        model=cfg.eval.api_model,
        max_tokens=cfg.eval.max_tokens,
        temperature=cfg.eval.temperature,
        system=messages[0]["content"],
        messages=messages[1:],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text")
    return raw, resp.usage.input_tokens, resp.usage.output_tokens


def run_baseline_inference(
    cfg: Config, mode: str, eval_set: list[DevExample], shots: list[DevExample],
    *, generate_fn=None,
) -> list[dict]:
    """Generate + parse predictions for one shot-mode over the frozen test set.

    ``generate_fn(messages) -> (raw, input_tokens, output_tokens)`` is injected in tests to run
    the assembly offline; in production it defaults to the lazy Anthropic client. Egress is
    hard-capped at ``eval.request_max_snippets`` — the single egress point of S4; the cap bounds
    the eval set (few-shot mode also sends the small fixed ``few_shot_examples`` train-head shots).
    """
    cap = cfg.eval.request_max_snippets
    if len(eval_set) > cap:
        raise ValueError(f"{len(eval_set)} snippets exceeds the egress cap ({cap})")

    n_shots = cfg.eval.few_shot_examples if mode == "few" else 0
    alias_index = load_tech_aliases()
    if generate_fn is None:
        client = _build_client()

        def generate_fn(messages):
            return _api_generate(client, cfg, messages)

    preds: list[dict] = []
    for ex in eval_set:
        messages = build_messages(ex, shots, n_shots=n_shots)
        t0 = time.perf_counter()
        raw, in_tok, out_tok = generate_fn(messages)
        latency = time.perf_counter() - t0
        parsed, valid = parse_output(raw, alias_index)
        preds.append({
            "offer_id": ex.offer_id, "valid": valid, "parsed": parsed,
            "latency_s": round(latency, 3),
            "input_tokens": in_tok, "output_tokens": out_tok,
        })
    return preds


def write_predictions(variant: str, preds: list[dict], pred_dir: Path = _PRED_DIR) -> Path:
    pred_dir.mkdir(parents=True, exist_ok=True)
    path = pred_dir / f"{variant}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    return path


def run_baselines(
    cfg: Config, *, processed_dir: Path = _PROCESSED, pred_dir: Path = _PRED_DIR,
    limit: int = 0, generate_fn=None,
) -> dict[str, list[dict]]:
    """Run every ``eval.shot_modes`` variant over the frozen test set; shots come from train head.

    Eval set = the frozen S2 test split (ADR-0003 "same frozen test set"); shots are the head of
    train (train/test are already offer-disjoint, so no leakage). Writes one predictions file per
    variant and returns them keyed by ``{model}__{mode}``.
    """
    train = _read_jsonl(processed_dir / "train.jsonl")
    test = _read_jsonl(processed_dir / "test.jsonl")
    shots = to_dev_examples(train[: cfg.eval.few_shot_examples])
    eval_recs = test[:limit] if limit else test
    eval_set = to_dev_examples(eval_recs)

    out: dict[str, list[dict]] = {}
    for mode in cfg.eval.shot_modes:
        preds = run_baseline_inference(cfg, mode, eval_set, shots, generate_fn=generate_fn)
        variant = f"{cfg.eval.api_model}__{mode}"
        write_predictions(variant, preds, pred_dir)
        out[variant] = preds
    return out


def main() -> None:
    cfg = load_config()
    out = run_baselines(cfg)
    for variant, preds in out.items():
        valid = sum(int(p["valid"]) for p in preds)
        print(f"[eval] {variant}: {len(preds)} predictions, {valid} valid JSON -> {_PRED_DIR}")


if __name__ == "__main__":
    main()

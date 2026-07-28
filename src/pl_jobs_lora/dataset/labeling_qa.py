"""Triangulated labeling-QA loop (S3; ADR-0005): propose from prose, sample, report.

Orchestration + I/O only — the agreement math is the pure ``dataset.agreement`` module and the
inference reuses the ADR-0001 probe (``probe.run_inference``, whose llama-cpp import is lazy). The
LLM (Bielik-1.5B few-shot) proposes labels from PROSE ONLY over the frozen S2 set; agreement vs the
platform gold is computed at full scale, and a seeded, disagreement-stratified sample is drawn for
human adjudication. Proposals are cached (replayable), the review queue and human gold are
gitignored (may echo prose), and only the numbers-only ``report.{json,md}`` are committable.

    python -m pl_jobs_lora.dataset.labeling_qa --propose   # Bielik few-shot over the frozen set
    python -m pl_jobs_lora.dataset.labeling_qa --sample     # draw the human-review queue
    python -m pl_jobs_lora.dataset.labeling_qa --arbiter    # pre-fill the queue via the API arbiter
    python -m pl_jobs_lora.dataset.labeling_qa --report     # report from the filled human gold
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from pl_jobs_lora.config import Config, ModelCandidate, load_config
from pl_jobs_lora.dataset import agreement
from pl_jobs_lora.dataset.agreement import AgreementReport
from pl_jobs_lora.dataset.collect import DevExample
from pl_jobs_lora.eval.prompt import build_messages, parse_output
from pl_jobs_lora.normalize import load_tech_aliases
from pl_jobs_lora.schema import JobPosting

_ROOT = Path(__file__).resolve().parents[3]
_PROCESSED = _ROOT / "data" / "processed"              # frozen S2 train/test (gitignored)
_OUT = _ROOT / "results" / "labeling_qa"
_PROPOSALS = _OUT / "proposals.jsonl"                  # gitignored: cached LLM proposals, replayed
_REVIEW = _OUT / "review_queue.jsonl"                  # gitignored: echoes prose, human fills it
_HUMAN_GOLD = _OUT / "human_gold.jsonl"                # gitignored: the adjudicated sample
_REPORT_JSON = _OUT / "report.json"                    # committable: numbers only
_REPORT_MD = _OUT / "report.md"                        # committable: numbers only

# The label fields carried through the QA (JobPosting-shaped; contract_types is kept for the human
# but not scored). Set fields default to [] and scalar fields to None in the empty template.
_GOLD_FIELDS = (
    "title", "seniority", "work_mode", "tech_expected", "tech_optional", "contract_types", "salary",
)
_SET_FIELDS = ("seniority", "work_mode", "tech_expected", "tech_optional", "contract_types")
# Half the sample is drawn from the disagreement stratum, over-weighting it vs its (low) base
# rate so the human's scarce time lands on contested labels — where gaps and LLM errors live.
_DISAGREEMENT_TARGET_SHARE = 0.5
# Arbiter (ADR-0005): a forced structured tool call bounds the output to the JobPosting schema; the
# result is then normalized through the shared parser, exactly like the probe (ADR-0003 fairness).
_ARBITER_MAX_TOKENS = 2048
_ARBITER_TOOL = "emit_job_posting"


def _read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_records(processed_dir: Path = _PROCESSED) -> list[dict]:
    """The frozen S2 set (train + test); each record has offer_id, url, pub_date, prose, gold."""
    return _read_jsonl(processed_dir / "train.jsonl") + _read_jsonl(processed_dir / "test.jsonl")


def to_dev_examples(records: list[dict]) -> list[DevExample]:
    return [DevExample(**r) for r in records]


def _empty_gold() -> dict:
    return {f: ([] if f in _SET_FIELDS else None) for f in _GOLD_FIELDS}


# -- the three label legs (agreement operates on flat {offer_id, ...fields} records) ---------------

def platform_leg(records: list[dict]) -> list[dict]:
    return [{"offer_id": r["offer_id"], **r["gold"]} for r in records]


def llm_leg(proposals: list[dict]) -> list[dict]:
    """Flatten cached proposals into label records; ``valid`` carries JSON-validity onward."""
    return [
        {"offer_id": p["offer_id"], "valid": p.get("valid", False), **(p.get("parsed") or {})}
        for p in proposals  # extra keys (offer_id/valid) are ignored by the field scorer
    ]


def human_leg(rows: list[dict]) -> list[dict]:
    """Accept either review-queue rows (nested ``human_gold``) or flat hand-authored records."""
    out = []
    for row in rows:
        hg = row.get("human_gold", row)
        out.append({"offer_id": row["offer_id"], **{k: hg[k] for k in _GOLD_FIELDS if k in hg}})
    return out


# -- propose: local Bielik few-shot over the frozen set (reuses the probe) -------------------------

def _proposer_candidate(cfg: Config) -> ModelCandidate:
    for c in cfg.models:
        if c.key == cfg.labeling_qa.proposer:
            return c
    raise ValueError(f"proposer {cfg.labeling_qa.proposer!r} not in models.candidates")  # defensive


def split_shots_eval(
    train: list[dict], test: list[dict], n_shots: int,
) -> tuple[list[dict], list[dict]]:
    """Shots come from the head of train and are EXCLUDED from the proposed (eval) set."""
    return train[:n_shots], train[n_shots:] + test


def propose(cfg: Config, *, processed_dir: Path = _PROCESSED, limit: int = 0) -> list[dict]:
    """Run the proposer over the frozen set (prose-only) and cache proposals; needs gguf."""
    from pl_jobs_lora import probe  # local: probe.run_inference lazily imports llama-cpp

    train = _read_jsonl(processed_dir / "train.jsonl")
    test = _read_jsonl(processed_dir / "test.jsonl")
    shots_recs, eval_recs = split_shots_eval(train, test, cfg.probe.few_shot_examples)
    if limit:
        eval_recs = eval_recs[:limit]
    preds = probe.run_inference(
        _proposer_candidate(cfg), cfg.labeling_qa.proposer_mode,
        to_dev_examples(eval_recs), to_dev_examples(shots_recs), cfg,
    )
    _write_jsonl(_PROPOSALS, preds)
    return preds


# -- sample: seeded, disagreement-stratified draw for human adjudication ---------------------------

def select_sample(
    offer_ids: list[str], disagreements, n: int, seed: int, strategy: str,
) -> list[str]:
    """PURE: pick ``n`` offer ids to hand-check; ``stratified`` over-weights disagreements."""
    rng = random.Random(seed)
    ids = sorted(set(offer_ids))  # sort first → deterministic regardless of input order
    n = min(n, len(ids))
    if strategy != "stratified":
        return sorted(rng.sample(ids, n))

    dis = sorted(set(disagreements) & set(ids))
    agree = sorted(set(ids) - set(dis))
    take_dis = min(len(dis), round(n * _DISAGREEMENT_TARGET_SHARE))
    take_agree = min(len(agree), n - take_dis)
    take_dis = n - take_agree  # backfill from disagreements if the agreement stratum is short
    chosen = rng.sample(dis, take_dis) + rng.sample(agree, take_agree)
    return sorted(chosen)


def build_review_queue(
    cfg: Config, sample_ids: list[str], records: list[dict], proposals: list[dict],
) -> list[dict]:
    """One row per sampled offer: prose + LLM/platform labels + disagreement flags + empty gold."""
    tol = cfg.scoring.salary_rel_tolerance
    recs_by = {r["offer_id"]: r for r in records}
    prop_by = {p["offer_id"]: (p.get("parsed") or {}) for p in proposals}
    rows = []
    for oid in sample_ids:
        gold = recs_by[oid]["gold"]
        llm = prop_by.get(oid, {})
        rows.append({
            "offer_id": oid,
            "prose": recs_by[oid]["prose"],
            "disagreement": agreement.field_disagreements(llm, gold, salary_rel_tolerance=tol),
            "llm": {k: llm.get(k) for k in _GOLD_FIELDS},
            "platform": {k: gold.get(k) for k in _GOLD_FIELDS},
            "human_gold": _empty_gold(),  # human fills; every contested field must be adjudicated
        })
    _write_jsonl(_REVIEW, rows)
    return rows


def build_sample(cfg: Config, *, processed_dir: Path = _PROCESSED) -> list[dict]:
    records = load_records(processed_dir)
    proposals = _read_jsonl(_PROPOSALS)
    tol = cfg.scoring.salary_rel_tolerance
    dis = agreement.disagreeing_ids(
        llm_leg(proposals), platform_leg(records), salary_rel_tolerance=tol,
    )
    sample_ids = select_sample(
        [p["offer_id"] for p in proposals], dis,
        cfg.labeling_qa.human_sample_size, cfg.labeling_qa.sampling_seed, cfg.labeling_qa.sampling,
    )
    return build_review_queue(cfg, sample_ids, records, proposals)


# -- arbiter: independent API first pass that pre-fills the human gold (ADR-0005) ------------------

def _build_arbiter_client():
    """Lazy Anthropic client (optional ``api`` extra); the key comes from ANTHROPIC_API_KEY."""
    import anthropic

    return anthropic.Anthropic()


def _arbiter_label(client, cfg: Config, prose: str, alias_index: dict[str, str]):
    """One arbiter call: forced structured tool output, normalized via the shared parser.

    Reuses the same prompt as the probe and S4 baselines (ADR-0003) and forces a tool call whose
    schema is ``JobPosting`` — so the output is schema-bounded without any sampling params, which
    Opus 4.8 rejects. Returns ``(normalized dict | None, valid)`` from ``parse_output``.
    """
    ex = DevExample(offer_id="", url="", pub_date=None, prose=prose, gold=_empty_gold())
    messages = build_messages(ex, [], n_shots=0)
    resp = client.messages.create(
        model=cfg.labeling_qa.arbiter,
        max_tokens=_ARBITER_MAX_TOKENS,
        system=messages[0]["content"],
        messages=messages[1:],
        tools=[{
            "name": _ARBITER_TOOL,
            "description": "Return the structured extraction of the posting.",
            "input_schema": JobPosting.model_json_schema(),
        }],
        tool_choice={"type": "tool", "name": _ARBITER_TOOL},
    )  # no temperature/top_p/top_k — Opus 4.8 rejects sampling params
    payload = next((b.input for b in resp.content if b.type == "tool_use"), None)
    if payload is None:
        return None, False
    return parse_output(json.dumps(payload, ensure_ascii=False), alias_index)


def arbiter_prefill(cfg: Config, rows: list[dict], *, label_fn=None) -> list[dict]:
    """Pre-fill each review row's ``human_gold`` with an independent API arbiter (ADR-0005).

    A distinct model from the Bielik proposer (``claude-opus-4-8``): it reads the same PROSE and
    proposes labels, so the human adjudicates a first pass rather than a blank form. Egress is
    hard-capped at ``arbiter_max_snippets`` PII-free snippets — the single egress point of S3. Pass
    ``label_fn`` (prose -> ``(parsed, valid)``) to exercise the assembly offline, without a network.
    """
    cap = cfg.labeling_qa.arbiter_max_snippets
    if len(rows) > cap:
        raise ValueError(f"{len(rows)} snippets exceeds the arbiter_max_snippets={cap} egress cap")
    if label_fn is None:
        client = _build_arbiter_client()
        alias_index = load_tech_aliases()

        def label_fn(prose: str):
            return _arbiter_label(client, cfg, prose, alias_index)

    out = []
    for row in rows:
        parsed, valid = label_fn(row["prose"])
        gold = _empty_gold()
        for k, v in (parsed or {}).items():
            if k in _GOLD_FIELDS:
                gold[k] = v
        out.append({**row, "human_gold": gold, "arbiter_valid": valid})
    return out


def run_arbiter(cfg: Config) -> list[dict]:
    """Load the review queue, pre-fill it via the arbiter, and write it back in place."""
    rows = _read_jsonl(_REVIEW)
    filled = arbiter_prefill(cfg, rows)
    _write_jsonl(_REVIEW, filled)
    return filled


# -- report: three agreement legs + triangulation over the filled human gold -----------------------

def _validate_human_gold(rows: list[dict]) -> None:
    """Fail clearly on a malformed hand-edited row rather than silently mis-scoring it."""
    from pydantic import ValidationError

    from pl_jobs_lora.schema import JobPosting

    for row in rows:
        hg = row.get("human_gold", row)
        fields = {k: hg[k] for k in _GOLD_FIELDS if k in hg and hg[k] is not None}
        try:
            JobPosting(**fields)
        except ValidationError as e:
            oid = row.get("offer_id")
            raise ValueError(f"human_gold row {oid!r} is not a valid JobPosting: {e}") from e


def build_report(
    cfg: Config, *, processed_dir: Path = _PROCESSED,
    proposals_path: Path = _PROPOSALS, human_path: Path = _HUMAN_GOLD,
) -> AgreementReport:
    tol = cfg.scoring.salary_rel_tolerance
    records = load_records(processed_dir)
    llm = llm_leg(_read_jsonl(proposals_path))
    plat = platform_leg(records)
    llm_vs_platform = agreement.pairwise(llm, plat, salary_rel_tolerance=tol)

    llm_vs_human = platform_vs_human = triangulation = None
    n_sample = 0
    if human_path.exists():
        rows = _read_jsonl(human_path)
        _validate_human_gold(rows)
        human = human_leg(rows)
        n_sample = len(human)
        llm_vs_human = agreement.pairwise(llm, human, salary_rel_tolerance=tol)
        platform_vs_human = agreement.pairwise(plat, human, salary_rel_tolerance=tol)
        triangulation = agreement.triangulate(llm, plat, human, salary_rel_tolerance=tol)

    metadata = {
        "proposer": cfg.labeling_qa.proposer,
        "proposer_mode": cfg.labeling_qa.proposer_mode,
        "arbiter": cfg.labeling_qa.arbiter,
        "n_full": llm_vs_platform.n,
        "n_sample": n_sample,
        "sampling": cfg.labeling_qa.sampling,
        "sampling_seed": cfg.labeling_qa.sampling_seed,
        "note": "LLM leg excludes the few-shot shots; n_full is the proposed set.",
    }
    return AgreementReport(
        metadata=metadata, llm_vs_platform=llm_vs_platform,
        llm_vs_human=llm_vs_human, platform_vs_human=platform_vs_human, triangulation=triangulation,
    )


def write_report(report: AgreementReport, out_dir: Path = _OUT) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(report.render_markdown(), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Triangulated labeling-QA loop (S3; ADR-0005).")
    ap.add_argument("--propose", action="store_true", help="run the proposer over the frozen set")
    ap.add_argument("--sample", action="store_true", help="draw the seeded human-review queue")
    ap.add_argument("--arbiter", action="store_true", help="pre-fill the queue via the API arbiter")
    ap.add_argument("--report", action="store_true", help="build the agreement report")
    ap.add_argument("--limit", type=int, default=0, help="cap proposals for a smoke run (0=all)")
    args = ap.parse_args()

    cfg = load_config()
    if args.propose:
        preds = propose(cfg, limit=args.limit)
        print(f"[qa] proposed {len(preds)} records -> {_PROPOSALS}")
    if args.sample:
        rows = build_sample(cfg)
        print(f"[qa] review queue {len(rows)} rows -> {_REVIEW}")
        print("[qa] fill each human_gold, then save the file as human_gold.jsonl")
    if args.arbiter:
        filled = run_arbiter(cfg)
        print(f"[qa] arbiter pre-filled {len(filled)} rows -> {_REVIEW}")
        print("[qa] adjudicate every contested cell, then save as human_gold.jsonl")
    if args.report:
        report = build_report(cfg)
        write_report(report)
        m = report.metadata
        print(f"[qa] report n_full={m['n_full']} n_sample={m['n_sample']} -> {_REPORT_MD}")


if __name__ == "__main__":
    main()

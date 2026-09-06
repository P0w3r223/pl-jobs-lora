# CLAUDE.md — pl-jobs-lora

Guidance for Claude Code (and any contributor) working in this repository.

## What this project is

Portfolio project **P4** (stage 2). A **QLoRA fine-tune** of a small open Polish LLM
(~1.5B) that turns Polish IT job-posting **prose → structured JSON** (seniority,
technologies, work mode, salary), trained on a **self-built dataset** drawn from the
sibling project `it-job-radar`, and compared honestly against zero-shot / few-shot API
baselines on **accuracy × cost × latency**. The point is the methodology — own dataset,
label-quality QA, apples-to-apples evaluation — not "I called an API".

## Architecture

Every module below exists and is tested. This map used to be headed "(target)" with stage
markers against the packages, which read as a plan; the repository overtook it.

```
src/pl_jobs_lora/
  vocab.py        # canonical seniority/work-mode/salary vocabulary (vendored from it-job-radar)
  schema.py       # pydantic JobPosting — the extraction target + prompt schema
  normalize.py    # vendored it-job-radar normalization (eval fairness)
  config.py       # configs/config.yaml -> frozen dataclasses; no hardcoded values
  tracking.py     # MLflow wiring; MLFLOW_TRACKING_URI env wins over config
  resume.py       # every long producer resumes through this: a torn trailing line is dropped,
                  #   a record without `required_keys` is not a result, and `matches` refuses a
                  #   cache written under a different configuration
  probe.py        # base-model selection probe (Bielik-1.5B vs Qwen2.5-1.5B)
  dataset/        # collect + build + time-based split; labeling QA + agreement report
  eval/           # metrics, PURE scorer (predictions x gold), API baselines, report
  train/          # qlora.py — hosted-GPU QLoRA fine-tune; pure SFT formatting testable locally
  inference/      # predict_hf (hosted GPU, base +/- adapter), predict_gguf (local CPU latency)
configs/config.yaml       # all knobs
requirements-train.txt    # GPU deps (transformers/peft/bitsandbytes) — hosted only, never local
docs/index.html           # the published page; guarded by tests/test_docs_page.py
results/eval/report.md    # the generated artifact the page and README quote
docs/decisions/           # ADRs
```

## Rules (do not violate)

- **Local vs hosted-GPU dependency split.** The local `.venv` does data building, scoring, and
  the CPU probe. The GPU/Linux training stack lives in `requirements-train.txt` and is installed
  **only on the hosted GPU** — never add it to `pyproject.toml` (ADR-0004).
- **The hosted GPU is Kaggle, not Colab.** ADR-0004 originally said "free Colab/Kaggle" and the
  notebook implemented Colab; its 2026-08-21 amendment records why that failed and what replaced
  it. `notebooks/train_qlora.ipynb` imports `kaggle_secrets`, and `train/qlora.py` says so in its
  first line. Anything written for Colab is written for the wrong platform.
- **Normalize before scoring.** Predictions and gold both pass through `normalize.py` so
  `ReactJS` vs `react` and the Polish `regular` → mid quirk are matches, not errors (ADR-0003).
- **The scorer is pure.** `(predictions, gold) → scores`, model-free and offline. Each variant
  emits `predictions/{variant}.jsonl`; scoring never loads a model.
- **Raw postings never leave the machine.** Capture prose at collection time; freeze the
  processed dataset on HF Hub; commit only prose-derived fields. No PII (drop `applying`).
  The one committed `prose` field is `data/fixtures/labeling_qa/`, whose five offers are
  invented rather than collected — the exemption and its reasoning are in ADR-0005 and in that
  directory's own README.
- **Time-based split, never random.** Train = older, test = newest (by publication date).
- **No hardcoded values.** Model IDs, thresholds, repos live in `configs/config.yaml`.
- **The page quotes; it never retypes.** Every figure `docs/index.html` prints has to be a figure
  a committed artifact prints — no rounding, no re-derivation. `tests/test_docs_page.py` carries
  the rule, because the portfolio-wide checker cannot see this repository's artifacts. The spec
  is `docs/audit/0007_divergence-and-the-page-spec.md` §5 in the private portfolio index, and it
  sweeps all twelve surfaces from there.

## Conventions

- English for code, comments, README, commits. Conventional Commits; one PR per session.
- Interpreter: `.venv/Scripts/python.exe` (Python 3.12). On Windows use `PYTHONIOENCODING=utf-8`
  for Polish characters.
- ruff (`E,F,I,UP,B,SIM,RUF`, line-length 100) + pytest green before commit.

## How to run

Two of these cost money or hours. They are marked, and nothing else here leaves the machine.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # Windows; local deps only
pytest
ruff check .

# data
.venv/Scripts/python -m pl_jobs_lora.dataset.run --collect --limit 20  # smoke: fetch 20
.venv/Scripts/python -m pl_jobs_lora.dataset.run                       # replay cache, rebuild
.venv/Scripts/python -m pl_jobs_lora.dataset.labeling_qa --report      # agreement + triangulation

# evaluation
.venv/Scripts/python -m pl_jobs_lora.eval.run --baselines   # PAID, network — the API baselines
.venv/Scripts/python -m pl_jobs_lora.eval.run --report      # score every predictions file, offline
.venv/Scripts/python -m pl_jobs_lora.probe --rescore        # re-score stored predictions, no model

# inference
.venv/Scripts/python -m pl_jobs_lora.inference.predict_gguf --mode both  # ~5 h CPU, resumable

# hosted GPU (Kaggle), not local
python -m pl_jobs_lora.train.qlora --push
python -m pl_jobs_lora.inference.predict_hf --base --lora
```

## Code intelligence

One index exists over this repo:

- `.code-review-graph/` — its MCP server is declared in **this repository's** `.mcp.json`, so it
  loads when Claude Code runs with this directory as the working directory, and is simply absent
  when the session started in the private portfolio index one level up. When its tools are
  missing the CLI still works: `uvx code-review-graph <command>`.

**The index has no hook**, so it is only as fresh as the last manual update — and a graph that
predates the work you are looking at will answer confidently about code that is gone. Run
`uvx code-review-graph update` before trusting it on a question about recent changes.

Grep, Glob and Read stay correct whenever the question is about text rather than structure, or
when neither index is available.

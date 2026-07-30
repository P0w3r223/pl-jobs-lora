# CLAUDE.md — pl-jobs-lora

Guidance for Claude Code (and any contributor) working in this repository.

## What this project is

Portfolio project **P4** (stage 2). A **QLoRA fine-tune** of a small open Polish LLM
(~1.5B) that turns Polish IT job-posting **prose → structured JSON** (seniority,
technologies, work mode, salary), trained on a **self-built dataset** drawn from the
sibling project `it-job-radar`, and compared honestly against zero-shot / few-shot API
baselines on **accuracy × cost × latency**. The point is the methodology — own dataset,
label-quality QA, apples-to-apples evaluation — not "I called an API".

## Architecture (target)

```
src/pl_jobs_lora/
  vocab.py        # canonical seniority/work-mode/salary vocabulary (vendored from it-job-radar)
  schema.py       # pydantic JobPosting — the extraction target + prompt schema
  normalize.py    # vendored it-job-radar normalization (eval fairness)
  config.py       # configs/config.yaml -> frozen dataclasses; no hardcoded values
  tracking.py     # MLflow wiring; MLFLOW_TRACKING_URI env wins over config
  dataset/        # collect + build + split (time-based) [S2 done]; labeling QA + agreement report [S3]
  eval/           # metrics, PURE scorer (predictions x gold), API baselines, report            [S4]
  train/          # qlora.py — Colab-only QLoRA fine-tune; pure SFT formatting testable locally [S5]
  inference/      # predict_hf (Colab GPU, base +/- adapter), predict_gguf (local CPU latency)  [S5]
  probe.py        # base-model selection probe (Bielik-1.5B vs Qwen2.5-1.5B)                     [S1]
configs/config.yaml       # all knobs
requirements-train.txt    # Colab-only GPU deps (transformers/peft/bitsandbytes) — never local
docs/decisions/           # ADRs
```

## Rules (do not violate)

- **Local vs Colab dependency split.** The local `.venv` does data building, scoring, and the
  CPU probe. The GPU/Linux training stack lives in `requirements-train.txt` and is installed
  **only on Colab** — never add it to `pyproject.toml` (ADR-0004).
- **Normalize before scoring.** Predictions and gold both pass through `normalize.py` so
  `ReactJS` vs `react` and the Polish `regular` → mid quirk are matches, not errors (ADR-0003).
- **The scorer is pure.** `(predictions, gold) → scores`, model-free and offline. Each variant
  emits `predictions/{variant}.jsonl`; scoring never loads a model.
- **Raw postings never leave the machine.** Capture prose at collection time; freeze the
  processed dataset on HF Hub; commit only prose-derived fields. No PII (drop `applying`).
- **Time-based split, never random.** Train = older, test = newest (by publication date).
- **No hardcoded values.** Model IDs, thresholds, repos live in `configs/config.yaml`.

## Conventions

- English for code, comments, README, commits. Conventional Commits; one PR per session.
- Interpreter: `.venv/Scripts/python.exe` (Python 3.12). On Windows use `PYTHONIOENCODING=utf-8`
  for Polish characters.
- ruff (`E,F,I,UP,B,SIM,RUF`, line-length 100) + pytest green before commit.

## How to run

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # Windows; local deps only
pytest
ruff check .
```

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes_tool` or `query_graph_tool` instead of Grep
- **Understanding impact**: `get_impact_radius_tool` instead of manually tracing imports
- **Code review**: `detect_changes_tool` + `get_review_context_tool` instead of reading entire files
- **Finding relationships**: `query_graph_tool` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview_tool` + `list_communities_tool`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
| ------ | ---------- |
| `detect_changes_tool` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context_tool` | Need source snippets for review — token-efficient |
| `get_impact_radius_tool` | Understanding blast radius of a change |
| `get_affected_flows_tool` | Finding which execution paths are impacted |
| `query_graph_tool` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes_tool` | Finding functions/classes by name or keyword |
| `get_architecture_overview_tool` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. No hooks installed — run `code-review-graph update` after code changes.
2. Use `detect_changes_tool` for code review.
3. Use `get_affected_flows_tool` to understand impact.
4. Use `query_graph_tool` pattern="tests_for" to check coverage.

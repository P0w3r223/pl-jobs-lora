# pl-jobs-lora

**A QLoRA fine-tune of a small Polish LLM that turns Polish IT job-posting prose into structured
JSON — compared honestly against zero-shot / few-shot API baselines on accuracy × cost × latency.**

Portfolio project **P4** (stage 2). Given the free-text of a Polish IT job posting, the model
extracts a structured record — seniority, required/optional technologies, work mode, and salary —
as validated JSON. The dataset is **built from the sibling project [`it-job-radar`](https://github.com/P0w3r223/it-job-radar)**,
labeled with a triangulated QA loop, and every model variant is scored by one shared harness. The
argument the project makes: *a 1.5B model I fine-tuned myself can rival a frontier API on this narrow
task at a fraction of the cost — and here is the measurement.*

> Status: **in progress (session 1 of ~5) — scaffold complete.** The extraction contract, the
> vendored normalization (for fair scoring), the config/tracking layer, and the design decisions are
> built and tested. The dataset, the evaluation harness, and the trained adapter land in the next
> sessions. Numbers below are shown as an **empty shape**, not invented values.

## Why it's built this way

- **Own dataset, not Kaggle.** The training data comes from a pipeline that has been collecting
  Polish IT postings for months — data nobody else has. The model input is the posting *prose*; the
  labels are triangulated from the platform's own structured fields, an LLM reading only the prose,
  and human adjudication of the disagreements. See [ADR-0002](docs/decisions/0002-dataset-and-labeling.md).
- **Honest, apples-to-apples evaluation.** Every variant — zero-shot API, few-shot API, the untuned
  base, the QLoRA model — is scored by the **same pure, model-free harness** on the **same frozen
  test set**, per field. See [ADR-0003](docs/decisions/0003-evaluation-methodology.md).
- **Cost thinking, not just accuracy.** The report answers "per 1000 postings: API $X vs a
  fine-tuned model at ~$0 — at what accuracy?" — the question an employer actually asks.
- **A defensible base-model choice.** Bielik-1.5B vs Qwen2.5-1.5B is decided by an empirical probe,
  not reputation. See [ADR-0001](docs/decisions/0001-base-model-selection.md).

## Architecture

```mermaid
flowchart LR
  TP[theprotocol posting] --> COL[collect: prose + platform labels + pub date]
  COL --> BLD[build: input prose / gold labels, leakage-guarded]
  BLD --> SPL[time-based split] --> HF[(frozen dataset on HF Hub)]
  HF --> BASE[zero/few-shot API + base model] --> PRED1[predictions.jsonl]
  HF --> TRAIN[QLoRA train on Colab] --> ADP[(adapter on HF)] --> PRED2[predictions.jsonl]
  PRED1 & PRED2 --> SCORE[pure scorer: per-field accuracy x cost x latency] --> REP[comparison table]
```

The **local `.venv`** builds the dataset, scores predictions, and runs the CPU base-model probe. The
**GPU/Linux training stack** (`transformers`/`peft`/`bitsandbytes`) lives in `requirements-train.txt`
and is installed **only on Colab** — the whole repo (dataset builder, harness, configs) is driven
locally; the Colab notebook only invokes its scripts. See [ADR-0004](docs/decisions/0004-training-infra.md).

## Install & test

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # Windows; local deps only, no GPU stack
pytest        # offline: no models, no network
ruff check .
```

## Evaluation (shape — populated in later sessions)

| Variant | JSON valid | Seniority F1 | Tech F1 | Work-mode F1 | Salary acc | Median cost | Median latency |
|---|---|---|---|---|---|---|---|
| zero-shot (API) | – | – | – | – | – | – | – |
| few-shot (API) | – | – | – | – | – | – | – |
| base (untuned) | – | – | – | – | – | – | – |
| **QLoRA (ours)** | – | – | – | – | – | – | – |

Metrics normalize both sides through the vendored it-job-radar functions (alias- and
Polish-quirk-aware), so the comparison is fair rather than penalizing paraphrases.

## Data & ethics

Raw postings are **never committed**: prose is captured at collection time (theprotocol postings
expire and the site strips pages for datacenter IPs), the processed dataset is frozen on HF Hub, and
only prose-derived fields leave the machine. Personal data (the `applying` block) is dropped.
Collection is a bounded, throttled sample — never the whole base. Attribution: theprotocol.it, reused
via `it-job-radar`.

## Design decisions

- [ADR-0001 — base-model selection by empirical probe](docs/decisions/0001-base-model-selection.md)
- [ADR-0002 — dataset construction + triangulated labeling QA](docs/decisions/0002-dataset-and-labeling.md)
- [ADR-0003 — per-field metrics with a pure scorer](docs/decisions/0003-evaluation-methodology.md)
- [ADR-0004 — Colab training, hosted MLflow, HF Hub artifacts](docs/decisions/0004-training-infra.md)
- [F6 — data-availability verdict](docs/research/f6-data-availability.md)

## License

MIT — see [LICENSE](LICENSE).

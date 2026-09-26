# Running the pipeline, step by step

### Base-model probe (ADR-0001)

Selects the QLoRA base by running both candidates (Qwen2.5-1.5B, Bielik-1.5B) zero- and few-shot
over a live dev slice on **local CPU via GGUF**, scored by the pure harness:

```bash
# CPU llama.cpp: on Windows use the prebuilt wheel (no compiler needed)
.venv/Scripts/python -m pip install llama-cpp-python --prefer-binary \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

.venv/Scripts/python -m pl_jobs_lora.probe --collect   # fetch slice, then probe both models
.venv/Scripts/python -m pl_jobs_lora.probe             # replay cached slice
.venv/Scripts/python -m pl_jobs_lora.probe --rescore   # re-score stored predictions (no model)
```

GGUFs download on demand; the slice, models, and per-run predictions/reports stay local (gitignored).
Both candidates use matched **Q8_0** quantization (Bielik ships no q4). Results land in
`results/probe/`; the decision and numbers are recorded in ADR-0001.

`--rescore` is the probe's counterpart to `eval.run --report`: it re-scores the **stored**
predictions with the current scorer, offline and without loading a model, so a change to the
metrics can be propagated to the published probe numbers without re-paying for inference. It was
added when the scoring correction below invalidated the salary columns of the ADR-0001 table — see
that ADR's 2026-08-21 amendment.

### Dataset build (ADR-0002)

Collects a bounded, throttled, spread sample of theprotocol offers, extracts **prose input** (titled
`jsonSections` only — the technologies widget is dropped as a label leak) + **platform-gold labels** +
publication date, filters the unusable, deduplicates reposts, and splits **by publication date**
(newest 20% → test) — never randomly:

```bash
.venv/Scripts/python -m pl_jobs_lora.dataset.run --collect --limit 20   # smoke: fetch 20, build, split
.venv/Scripts/python -m pl_jobs_lora.dataset.run --collect              # full ~800 collection + freeze
.venv/Scripts/python -m pl_jobs_lora.dataset.run                        # replay cached slice, rebuild
```

The collected slice and processed `data/processed/{train,test}.jsonl` + `manifest.json` stay local
(gitignored); freezing them on HF Hub (`--push`, needs `HF_TOKEN`) is deferred until the dataset repo
is provisioned. Everything downstream replays this frozen dataset — collect once, never re-scrape
(offers expire).

The current build: **710 records**, split **568 train / 142 test** by publication date — the three
figures `ADR-0002`, `configs/config.yaml` and `results/eval/report.json` carry. Label coverage for
salary is **31%**, which `ADR-0006` records and which is honestly sparse: it is often absent from the
posting.

*The per-field coverage is printed by nothing committed and was removed rather than corrected;
the fetch and dedupe counts are in `data/processed/manifest.json`, which the build writes and
`.gitignore` keeps local. A hand-typed figure with no instrument beside it only sets the next
staleness date, and the collection run itself now prints its drops — the other half of the same
repair. `dataset_manifest()` writes `collected`, `passed_filters`, `dropped_filters`,
`dropped_duplicates` and `records`; coverage is not among them, which the first edition of this
paragraph said it was.* Zero prose leaks the technologies widget (leakage guard). Cross-split leakage is prevented
by the **dedupe-before-split** ordering: train and test share **no offer id** and **no prose hash**.
The temporal cut is by publication date (train older, test newest); in this build it falls cleanly
between two postings ~14 min apart.

### Labeling QA (ADR-0005)

Before trusting the platform-gold labels, S3 triangulates three independent label sources and reports
their agreement. A local **Bielik-1.5B few-shot** proposer relabels every posting **from prose only**
(a small-model *prose-recoverability floor*, not a quality claim); its labels are compared to the
platform gold at full scale. A seeded, **disagreement-stratified 72-record sample** is then drawn for
human adjudication, and a distinct API arbiter (`claude-opus-4-8`) pre-fills each sample so the human
corrects contested cells instead of labeling from a blank form:

```bash
.venv/Scripts/python -m pl_jobs_lora.dataset.labeling_qa --propose   # Bielik few-shot over the set
.venv/Scripts/python -m pl_jobs_lora.dataset.labeling_qa --sample     # seeded, stratified queue
.venv/Scripts/python -m pl_jobs_lora.dataset.labeling_qa --arbiter    # pre-fill via the API arbiter
#     -> hand-adjudicate every contested cell, save as results/labeling_qa/human_gold.jsonl
.venv/Scripts/python -m pl_jobs_lora.dataset.labeling_qa --report     # agreement + triangulation
```

The report gives raw agreement, per-field F1 (reused scorer), and **Cohen's κ** for the categorical
fields, plus a per-field **triangulation** bucketing each cell into *all-agree / platform-gap-caught
(LLM==human≠platform) / LLM-error (platform==human≠LLM) / all-differ*. The agreement math
(`dataset/agreement.py`) and the sampler are **pure and offline**; only `dataset/labeling_qa.py`
touches the model, the network, and files. Proposals, the review queue, and human gold echo prose and
are **never committed** — only the numbers-only `results/labeling_qa/report.{json,md}` is versioned.
Egress is a single point, hard-capped at ≤80 PII-free prose snippets to the arbiter; the Anthropic
client is an optional `api` extra imported lazily, so the core install, tests, and CI stay offline
(CI exercises the agreement layer on synthetic `data/fixtures/labeling_qa/` only).

**Run status:** `--propose` and `--sample` have been run — 708/710 postings relabeled from prose, and
the seeded 72-record review queue is drawn. **Pending:** `--arbiter` (paid API) and the human
adjudication pass, then `--report`; all three are compute/manual steps outside this environment, not
code.

### Evaluation — API baselines + comparison report (ADR-0003)

S4 completes the `eval/` module: the **zero-/few-shot API baselines** and the **comparison report**
that scores every variant on accuracy × cost × latency. The baseline API model is a cheap frontier
model (`claude-haiku-4-5`); it reuses the **same prompt as the probe** and generates **plain text**
(not a forced tool call), so JSON validity stays a first-class metric the model can fail — the only
variable across variants is the model.

```bash
.venv/Scripts/python -m pl_jobs_lora.eval.run --baselines   # zero-/few-shot over test set (paid, network)
.venv/Scripts/python -m pl_jobs_lora.eval.run --report      # score every predictions file (offline)
.venv/Scripts/python -m pl_jobs_lora.eval.run --report --no-bootstrap   # skip the resampling section
```

`--baselines` runs the frozen test set through the API and writes
`results/eval/predictions/{model}__{mode}.jsonl` (per-call token counts + latency); it needs
`ANTHROPIC_API_KEY` and the optional `api` extra. `--report` is **pure and offline**: it scores each
predictions file with the ADR-0003 scorer, folds in API cost (tokens × list price, pulled from
`config.yaml`) and p50/p95 latency, and writes the numbers-only `results/eval/report.{json,md}`.
Variants without token counts — the local base/LoRA/GGUF runs (~$0 marginal) — report no cost, so the
same report merges the API baselines with predictions dropped in later from the hosted GPU. Predictions echo
prose and are **never committed**; only the report is versioned. Egress is hard-capped at
`eval.request_max_snippets` per run, and the Anthropic client is imported lazily (`api` extra), so the
core install, tests, and CI stay fully offline. The report answers the headline question: *does a 1.5B
local LoRA rival a frontier API on this task at a fraction of the cost/latency?*

Beyond the accuracy table, `--report` emits two sections that keep that answer interpretable: a
**failure taxonomy** (why each invalid prediction was invalid, with truncation separated from
malformed JSON) and **bootstrap intervals** with paired variant differences, so a gap on 142 records
is only called real when it survives resampling. Both are pure and offline; both are described under
Evaluation below and in [ADR-0003](decisions/0003-evaluation-methodology.md).

### Training — QLoRA fine-tune on a hosted GPU (ADR-0006)

S5 fine-tunes Bielik-1.5B with QLoRA and produces the base/adapter predictions that complete the
table. The method: **zero-shot completion SFT** — each record is *exactly* the zero-shot eval prompt
with the gold JSON as the target, prompt tokens masked so the model trains only on the JSON (learning
the stop token that the base model, which rambles, lacks). LoRA over **all linear layers**
(r=16/α=32, NF4 4-bit); an **inner temporal dev split** (newest ~10% of train) governs checkpoint
selection so the **142-record test set is scored exactly once** — the fairness guard, since the API
baselines get no tuning. Salary is honestly a data-availability ceiling (the leakage guard keeps it
out of the prose the model sees), so `tech_expected` F1 is the metric to watch.

Training needs a **Linux GPU** (bitsandbytes has no Windows/CPU build, and the local card is a 4 GB
Pascal), so it runs on a free **Kaggle** T4 via
[`notebooks/train_qlora.ipynb`](../notebooks/train_qlora.ipynb); the GPU stack lives in
`requirements-train.txt` and is installed **only there** (ADR-0004 and its 2026-08-21 amendment,
which records why Colab was abandoned). The whole repo is driven from the notebook — it only invokes
these scripts:

```bash
# on the hosted GPU, after `pip install -r requirements-train.txt`:
python -m pl_jobs_lora.train.qlora --push                 # fine-tune; push the adapter to HF
python -m pl_jobs_lora.inference.predict_hf --base --lora  # base (adapter off) + LoRA predictions
python -m pl_jobs_lora.eval.run --report                   # merge every variant into the table
```

`predict_hf` reuses the shared prompt + parser + greedy decoding (token cap = `eval.max_tokens`), and
loads the base and the adapter from the *same* 4-bit weights with only the adapter toggled — so the
adapter-vs-base comparison isolates one variable. Its predictions carry no token counts, so the report
prices them at ~$0 (local). The pure parts — SFT formatting, the temporal dev split, completion
masking — are tested on the local CPU `.venv`; the GPU orchestration is exercised only on the hosted GPU.
`inference/predict_gguf.py` optionally times the base on **local CPU via GGUF** for the latency column
(the adapter is not GGUF-converted). See [ADR-0006](decisions/0006-qlora-training-method.md).

```bash
.venv/Scripts/python -m pl_jobs_lora.inference.predict_gguf --mode both   # ~5 h CPU, resumable
.venv/Scripts/python -m pl_jobs_lora.inference.predict_gguf --mode zero --fresh
```

**The CPU runs are resumable**, because a full pass is roughly an hour few-shot and closer to four
zero-shot. Each prediction is appended and flushed as it is produced, so Ctrl-C costs at most the
records the OS had not yet written, and re-running the same command skips whatever is already on
disk (progress prints against the *whole* set, not the remaining slice). A process killed mid-write
leaves a torn final line; it is dropped and that one record recomputed. The mechanics live in
`resume.py` and are shared with the labeling-QA proposer, which needs the same guarantee.

One deliberate asymmetry: a row written **before the failure taxonomy existed** does not count as
done, so re-running is what backfills `failure`/`raw` onto variants measured earlier. Because that
discards the old rows, the previous file is copied to `*.jsonl.pre-taxonomy` first — regenerable,
but regenerating costs the hours this path exists to protect.

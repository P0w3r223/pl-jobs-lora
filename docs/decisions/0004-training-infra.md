# ADR-0004 — Training on Colab, hosted MLflow, HF Hub artifacts

Date: 2026-07-27
Status: accepted
Author: Piotr Cząstkiewicz

---

## Context

QLoRA training needs a Linux GPU (bitsandbytes has no usable Windows/CPU build), so training runs on
free Colab/Kaggle while the rest of the repo (dataset builder, eval harness, probe) is driven from
the local `.venv`. The tracker must be reachable from both local eval runs and ephemeral Colab
sessions. P1 mlops-car-price already standardized on MLflow.

## Options

- **A — Hosted MLflow, e.g. DagsHub free tier (chosen).** One tracker for local eval + Colab training
  curves; API parity with P1; survives Colab teardown. Needs an external account/token.
- **B — Weights & Biases.** Trivial from Colab, great curves, but a second tracker vs P1 →
  portfolio inconsistency.
- **C — Local MLflow file store only.** Rejected: Colab's `mlruns/` is ephemeral.

## Decision

Option A for cross-project consistency; fall back to B only if hosted MLflow proves fiddly. Adapter
artifacts go to **HF Hub** regardless.

## Consequences

- **Dependency split (enforced in code):** local `pyproject.toml` carries only data/eval/probe deps
  (pydantic, pandas, mlflow-client, hf-hub, rapidfuzz); the GPU stack
  (transformers/peft/bitsandbytes/accelerate/trl) lives in `requirements-train.txt`, installed only
  on Colab. Verified: `bitsandbytes` is absent from the local `.venv`.
- **Tracking is config, not code:** `tracking.resolve_tracking_uri()` lets `MLFLOW_TRACKING_URI` win
  over `configs/config.yaml` (`tracking_uri: null` by default = local store), so switching to the
  hosted endpoint is an env change.
- Open for S5: the actual hosted tracking URI + auth, and the HF adapter repo ID + token (config
  placeholders today).

## Implementation (S5)

The Colab-only trainer (`train/qlora.py`) and inference (`inference/predict_hf.py`) import the GPU
stack lazily inside their functions, so the modules stay importable and their pure parts (SFT
formatting, temporal dev split, completion masking, prediction assembly) are tested on the local
CPU `.venv` — `bitsandbytes` is never a local dependency (verified). `tracking.start_run` logs the
LoRA/quant config, per-epoch loss, and the final report; `MLFLOW_TRACKING_URI` still wins over the
config. The frozen dataset is pulled from HF into the gitignored `data/processed/` by
`dataset.hf_dataset.pull_dataset` (mirror of `push_dataset`), since a fresh Colab clone has no data.
The adapter is pushed to `hf.adapter_repo` (`P0w3r223/pl-jobs-lora-adapter`, created private on push,
like the dataset). The hosted MLflow URI + token remain runtime-supplied (env), still deferred.

## Amendment — 2026-08-21: the hosted GPU is Kaggle, not Colab

The ADR said "free Colab/Kaggle" and the notebook implemented Colab. Colab was attempted on
2026-08-04 and abandoned: the Bielik repo was gated for that account, and kernel restarts wiped the
gitignored `data/processed/`, so each retry began by re-pulling the dataset. The local machine is
not an option either — GTX 1050, 4 GB, Pascal — where a 4-bit 1.5B QLoRA run wants roughly 8 GB and
bitsandbytes is unreliable on that generation. `notebooks/train_qlora.ipynb` now targets **Kaggle**
(free T4, ~9 h per session, 30 h/week): tokens come from Kaggle Secrets rather than `getpass`, the
clone lands in `/kaggle/working`, and artefacts are collected from the notebook Output instead of
`google.colab.files`.

Nothing in the decision moves — the dependency split, the lazy GPU imports, HF Hub for artefacts and
env-supplied tracking are all platform-independent, which is why the switch cost one notebook and
zero lines of logic. The docstrings and README that said "Colab" now say **hosted GPU**: the
constraint they name is the Linux-GPU half of the split, not a vendor, and naming the vendor is what
let a platform change read as a code change.

Two consequences are Kaggle's alone. A fresh session starts with an empty `/kaggle/working`, so the
resume in `predict_hf`/`predict_gguf` protects a run within a session but not across two — carrying
one over means attaching the previous version's output as a dataset. And a T4 x2 accelerator leaves
`Trainer` seeing two GPUs, which wraps the model in DataParallel while the 4-bit weights sit where
`device_map` put them; the notebook pins `CUDA_VISIBLE_DEVICES=0` up front, since that failure
appears minutes into training rather than at load.

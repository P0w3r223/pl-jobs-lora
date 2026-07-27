# ADR-0004 — Training on Colab, hosted MLflow, HF Hub artifacts

Date: 2026-07-27
Status: accepted
Author: P0w3r223 + Claude

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

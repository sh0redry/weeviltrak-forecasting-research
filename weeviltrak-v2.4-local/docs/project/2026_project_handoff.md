# WeevilTrak 2026 Engineering Handoff

## Purpose

This is a concise operational handoff. For an explanatory introduction and a
recommended code/document reading path, start with the
[new engineer onboarding guide](new_engineer_onboarding_guide.md). This page
describes the maintained v2.4 workflow, where to look first, and what must be
treated as risk rather than assumed fact.

## Project in one paragraph

WeevilTrak predicts Annual Bluegrass Weevil timing from historical field
observations, weather-derived features, and location information. The current
pipeline is `v2.4`: LightGBM predicts signed days remaining for four canonical
business events, then an OOF-trained Ridge termination layer calibrates and
locks the operational event date. The maintained business outputs are Stage 1,
Phase 1, Stage 2, and Stage 3 / Phase 2.

## First-day reading and checks

1. Read the repository [`README.md`](../../README.md) and
   [`AGENTS.md`](../../AGENTS.md).
2. Read the [v2.4 architecture](../architecture/weeviltrak_model_process_overview.md)
   and [data risk register](../data/historical_data_contract_and_risks.md).
3. Set up Python 3.11 with Poetry; see [`manual/INSTALL.md`](../../manual/INSTALL.md).
4. Run the safe local test suite: `poetry run pytest tests/ -q`.
5. Before any external operation, read the [training and release runbook](../operations/v24_training_release_runbook.md).

## Current production facts

| Topic | Current rule |
| --- | --- |
| Default config | `app/config/weeviltrak_v2.4.yml` |
| Model | LightGBM Base Layer + learned Ridge termination layer |
| Training target | `signed_days_to_event` |
| Canonical model IDs | `1` Stage 1, `4` Phase 1, `2` Stage 2, `3` Stage 3 / Phase 2 |
| Training window | Recent 3 years, with direct-label coverage gate |
| Phase 1 | Direct model target; labels are historical paired Stage 1/2 midpoints and eligible quality-controlled observed Phase 1 labels |
| Phase 2 | Business alias of Stage 3 / Phase 2, canonical ID `3` |
| Manual observations | 11 clear mentor-provided events injected at runtime; never written to Redshift |
| Release mode | 2027 client release is inference-only during season; daily weather updates do not retrain artifacts |

Do not use `v2.5` for new artifacts. It is retained only for compatibility.

## Working modes

| Mode | Use | Main surface |
| --- | --- | --- |
| Local development | Unit/integration tests and code changes | `app/`, `tests/` |
| Training | Build and validate candidate Base + termination artifacts | `app/pipeline/train_predict.py` and runbook |
| Prediction-only | Load an approved artifact and generate current outputs | `app/pipeline/predict_only.py` |
| Backtesting | Evaluate historical trajectories and termination decisions | `app/pipeline/backtesting.py`, versioned caches/notebooks |
| Client delivery | Local pickle inference using immutable bundle | `manual/` |

## Key locations

| Path | Why it matters |
| --- | --- |
| `app/config/weeviltrak_v2.4.yml` | Production policy, feature list, target IDs, artifact/output paths |
| `app/services/canonical_events.py` | Stable Stage/Phase mapping and Phase 1 midpoint labels |
| `app/services/manual_events.py` | Curated runtime-only observations |
| `app/services/data_preparation_service.py` | Redshift pull, weather processing, feature matrix creation |
| `app/pipeline/train_predict.py` | Train/predict orchestration and OOF termination training |
| `app/pipeline/termination.py` | Learned termination features, training, and final business output logic |
| `scripts/release/create_v24_2027_release.py` | Immutable release-bundle creation |
| `manual/` | Customer-facing inference contract and deployment guidance |

## High-priority risks

- Redshift is not a clean one-to-one mirror of the historical Excel delivery.
  Confirmed date swaps exist in specific 2020, 2023, and 2024 source batches.
- `12/31` entries are placeholders, not observed event dates.
- Raw Stage IDs change meaning after the 2026 Phase transition; always use
  canonicalization.
- The recent 3-year window may not have enough direct labels for every output;
  the coverage gate is intentionally allowed to block retraining.
- Weather/S3/Redshift operations require credentials and can read or publish
  external data. Tests must not depend on them unless explicitly integration-scoped.

See [open risks and decisions](open_risks_and_decisions.md) for current
evidence and next actions.

## Handoff boundary

`outputs/` is generated local evidence, not source of truth. Never overwrite an
approved artifact or alter Redshift merely to make a notebook look consistent.
Retain source snapshots, manifests, and versioned results for every release.

# WeevilTrak Agent Context

This file is the primary working context for AI coding agents in this
repository.  It should describe the current codebase, not only the historical
plan.

## Project Summary

WeevilTrak is a Python ML pipeline for predicting boll weevil lifecycle timing
from historical pest observations and gridded weather features.

The app is not primarily a FastAPI service right now; the maintained surface is
the ML pipeline under `app/`, plus notebooks, scripts, tests, and local outputs.

New engineers should read `docs/project/2026_project_handoff.md`,
`docs/data/historical_data_contract_and_risks.md`, and
`docs/operations/v24_training_release_runbook.md` after this file. Historical
plans and investigations are evidence, not default operating instructions.

## Current Default Model

Use `app/config/weeviltrak_v2.4.yml` by default unless the task explicitly asks
for an older version.

`v2.4` is the current canonical Stage/Phase pipeline:

- model type: LightGBM + learned termination
- target: `signed_days_to_event`
- direct model targets:
  - Stage 1
  - Phase 1
  - Stage 2
  - Stage 3 / Phase 2
- business output:
  - Stage 1
  - Phase 1
  - Stage 2
  - Stage 3 / Phase 2
- Phase 1 is a direct model target whose training labels are:
  - `midpoint(observed Stage 1, observed Stage 2)`
- output prefix:
  - `weeviltrak_data/api-testing/v2.4`
- model artifact paths:
  - `model_testing_2025/model_weeviltrak_lgbm_v2_4.pkl`
  - `model_testing_2025/model_weeviltrak_termination_ridge_v2_4.pkl`
- `training_window_years` is intentionally set to `3`. Older observations are
  excluded because their collection process is less reliable. Before replacing
  an artifact, v2.4 checks direct-label coverage for every model stage; if the
  recent window is insufficient, retraining is blocked and the last approved
  artifact should be used through the prediction-only pipeline.

Older configs remain for comparison:

| Version | Status | Notes |
| --- | --- | --- |
| `v2.1` | legacy baseline | Random Forest, `days_to_event` |
| `v2.2` | previous baseline | LightGBM, signed target, monotonic constraints, zero-day termination |
| `v2.3` | previous default | LightGBM feature upgrade with DOY/CDD and 14-day rolling window |
| `v2.4` | current default | canonical Stage/Phase output, direct Phase 1 midpoint labels, manual curated events, OOF Ridge termination, 3-year coverage gate |
| `v2.5` | deprecated | compatibility config only; do not use for new artifacts |

See `docs/model_versions.md`.

## Stage/Phase Semantics

Raw Redshift `stage_id` values are not stable business targets after the 2026
data transition.  Use `app/services/canonical_events.py` for canonical mapping.

Canonical training labels in `v2.4`:

- raw `Stage 1` -> canonical `stage_id = 1`
- raw `Stage 2` -> canonical `stage_id = 2`
- raw `Stage 3` -> canonical `stage_id = 3`
- raw `Phase 2` -> canonical `stage_id = 3` (`Stage 3 / Phase 2`)
- `Phase 1` -> canonical `stage_id = 4`, generated only from paired Stage 1/2 midpoint labels
- placeholder dates such as `12-31` are excluded

Do not create fake Stage 1/2 labels from observed Phase 1.  Phase 1 is a
derived midpoint training label and a direct model target.

## Curated Manual Events

`v2.4` also includes 11 mentor-provided clear single-date observations from
`notebooks/experiments/manual_stage_data_validation_report.ipynb`.

The rows are hardcoded in `app/services/manual_events.py` and are appended after
Redshift pull/canonicalization when this config block is enabled:

```yaml
manual_events:
  enabled: true
  include_in_training: true
  include_in_validation: true
```

These rows do not modify Redshift, but at runtime they are treated like normal
labeled observations for weather pulls, feature construction, training, and
validation.  Missing, ambiguous, and interval manual observations are kept out
of direct training labels.

Important safety rule: manual events are still filtered by `today`, so
backtesting does not see future manual labels.

## Repository Layout

```text
app/
  config/                 YAML model/version configs
  models/                 BaseModelManager, RF/LGBM managers, factory
  pipeline/               train/predict, backtesting, hindcast, evaluation, termination
  services/               Redshift, weather/feature prep, canonical events, manual events
archive/                  retired modules, old notebooks, legacy scripts
docs/                     design notes and handoff docs
notebooks/
  production/             operational notebooks
  experiments/            exploratory analysis notebooks
outputs/                  local logs, caches, validation outputs, run artifacts
scripts/                  backtesting, hindcast, validation, weather utilities
tests/                    pytest tests
```

`outputs/` is local run output and should not be treated as source of truth
unless the user asks to inspect generated artifacts.

## Main Data Flow

```text
Redshift pest events
  -> DataPreparationService.pull_weevil_data()
  -> canonical Stage/Phase mapping
  -> optional curated manual event injection
  -> griddedweather weather pull/cache via S3/Redshift
  -> process_weather_data()
  -> prepare_training_data()
  -> model_manager.train()
  -> prediction + learned termination
  -> v2.4 business output mapping, including direct Phase 1
  -> local outputs and/or S3
```

## Core Components

### Config

- `app/config/weevilltrak_config.py`
- `app/config/weeviltrak_v2.4.yml`

Always prefer config-driven behavior over hardcoded version checks.

### Services

- `app/services/database_service.py`
  - Redshift connection wrapper.
- `app/services/data_preparation_service.py`
  - Pulls weevil data.
  - Pulls and normalizes weather data through `griddedweather`.
  - Builds annual cumulative and rolling weather features.
  - Prepares train/test matrices.
  - Adds post-event training rows.
- `app/services/canonical_events.py`
  - Stable Stage/Phase mapping and Phase 1 output derivation.
- `app/services/manual_events.py`
  - Runtime injection of the 11 curated manual observations.

### Models

- `app/models/base.py`
- `app/models/model_manager.py`
- `app/models/factory.py`

Use:

```python
from app.models.factory import create_model_manager

model_manager = create_model_manager(config, s3_manager)
pred = model_manager.predict(X)
```

Do not call the raw sklearn/LightGBM estimator directly.

### Pipelines

- `app/pipeline/base.py`
  - Shared service initialization.
- `app/pipeline/train_predict.py`
  - Main train + predict workflow.  Default script path uses `v2.4`.
- `app/pipeline/predict_only.py`
  - Load saved model and predict.
- `app/pipeline/backtesting.py`
  - Walk-forward validation.
- `app/pipeline/hindcast.py`
  - Historical what-if prediction utility.
- `app/pipeline/termination.py`
  - Learned termination / post-processing helpers.
- `app/pipeline/model_eval.py`
  - Evaluation metrics and visualizations.

## Environment

The project uses Python 3.11 and Poetry.

```bash
poetry install
poetry env info
```

Common environment variables:

```text
S3_BUCKET_NAME=sps-ds-bucket
REDSHIFT_HOST=<host>
REDSHIFT_USER=<user>
REDSHIFT_PASSWORD=<password>
REDSHIFT_DATABASE=<database>
REDSHIFT_PORT=5439
```

Some code also accepts `DATABASE_HOST`, `DATABASE_PORT`, and `DATABASE_NAME`.

For notebooks, create a Jupyter kernel from the Poetry environment:

```bash
poetry run python -m ipykernel install --user \
  --name weeviltrak-poetry \
  --display-name "WeevilTrak Poetry"
```

## Common Commands

```bash
# Install
poetry install

# Train + predict with current v2.4 default
poetry run python -m app.pipeline.train_predict

# Prediction only
poetry run python -m app.pipeline.predict_only \
  --config-path app/config/weeviltrak_v2.4.yml \
  --test-date 2026-03-02

# Backtesting
poetry run python -m app.pipeline.backtesting

# Hindcast
poetry run python -m app.pipeline.hindcast \
  --config-path app/config/weeviltrak_v2.4.yml \
  --target-date 2025-03-01

# Tests
poetry run pytest tests/ -v
```

Quality checks:

```bash
poetry run black app/ tests/ scripts/
poetry run isort app/ tests/ scripts/
poetry run flake8 app/ tests/
poetry run mypy app/
```

Current pytest suite is active; do not describe tests as empty.

## External Data

- Redshift source table:
  - `europe_dna.europe_dna_sps_weeviltrak`
- S3 bucket:
  - `sps-ds-bucket`
- Weather cache:
  - `s3://sps-ds-bucket/gridded-weather/cache/10by10/`
- Raw WeevilTrak S3 data inspected in prior work:
  - `s3://sps-ds-bucket/weeviltrak_data/raw_data/`

Network/S3/Redshift operations may require credentials and should not be
assumed available in tests.

### Data-quality safety

- `12/31` records are placeholders, not observed events.
- Excel and Redshift are not one-to-one; confirmed date swaps exist in specific
  2020/2023/2024 source batches.
- Do not bulk-correct Redshift dates from notebook matching alone. Preserve
  provenance and follow `docs/data/historical_data_contract_and_risks.md`.

## Output Conventions

Local outputs should go under:

- `outputs/backtesting/<version>/`
- `outputs/hindcast/<version>/`
- `outputs/business_checks/`
- `outputs/validation/`
- `outputs/cache/`
- `outputs/logs/`

Versioned S3 outputs should use the YAML `output_path`.

## Development Rules for Agents

- Preserve user changes.  Check `git status --short` before broad edits.
- Use `rg`/`rg --files` for search.
- Prefer `apply_patch` for source edits.
- Do not run destructive git commands unless the user explicitly asks.
- Do not hardcode credentials.
- Do not silently change older configs when adding a new modeling behavior.
- Keep version semantics clear.  A behavior-changing pipeline update usually
  deserves a new config version.
- Use primary app modules instead of duplicating notebook logic.
- Notebooks are useful evidence, but production logic belongs in `app/` with
  tests.

## High-Value Tests

Useful targeted tests when touching the current model flow:

```bash
poetry run pytest tests/unit/services/test_canonical_events.py -q
poetry run pytest tests/unit/models/test_learned_termination.py -q
poetry run pytest tests/unit/services/test_post_event_zero_training.py -q
poetry run pytest tests/unit/pipeline/test_signed_target_outputs.py -q
poetry run pytest tests/integration/test_hindcast_pipeline.py -q
```

Run the full suite before handing off production-facing changes:

```bash
poetry run pytest tests/ -q
```

## Important Files

| File | Purpose |
| --- | --- |
| `app/config/weeviltrak_v2.4.yml` | Current default model config |
| `app/services/canonical_events.py` | Stage/Phase canonicalization and Phase 1 derivation |
| `app/services/manual_events.py` | 11 curated manual observations |
| `app/services/data_preparation_service.py` | Data pull, weather processing, train/test prep |
| `app/pipeline/train_predict.py` | Main production train + predict pipeline |
| `app/pipeline/backtesting.py` | Walk-forward validation |
| `app/pipeline/hindcast.py` | Historical what-if prediction |
| `app/models/model_manager.py` | RF/LGBM training, prediction, serialization |
| `app/models/factory.py` | Model manager factory |
| `app/services/database_service.py` | Redshift connector |
| `docs/model_versions.md` | Version semantics |
| `README.md` | User-facing quickstart |

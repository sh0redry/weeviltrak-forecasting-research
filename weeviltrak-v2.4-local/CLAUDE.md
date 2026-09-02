# CLAUDE.md

This is the Claude-facing project context for `api-python-weeviltrak`.  It is
kept aligned with `AGENTS.md`, but is written as a compact English handoff.

## Current Repository State

WeevilTrak is a Python 3.11 + Poetry ML pipeline for predicting boll weevil
lifecycle timing from historical pest observations and gridded weather
features.

The maintained application surface is the ML pipeline under `app/`.  Notebooks
are used for exploration and reporting; production logic should live in `app/`
with tests.

For current onboarding and operational boundaries, also read:

- `docs/project/2026_project_handoff.md`
- `docs/data/historical_data_contract_and_risks.md`
- `docs/operations/v24_training_release_runbook.md`

Historical plans and investigations are evidence, not current operating
instructions unless a document explicitly says it is current.

## Default Model Version

Use `app/config/weeviltrak_v2.4.yml` by default.

`v2.4` is the current production-facing Stage/Phase pipeline:

- LightGBM primary model
- learned Ridge termination model
- target: `signed_days_to_event`
- direct model targets: Stage 1, Phase 1, Stage 2, Stage 3 / Phase 2
- business output: Stage 1, Phase 1, Stage 2, Stage 3 / Phase 2
- Phase 1 labels are `midpoint(observed Stage 1, observed Stage 2)` and are predicted directly
- output prefix: `weeviltrak_data/api-testing/v2.4`
- a recent 3-year training window for both LightGBM and learned termination

Older observations are intentionally excluded because their collection process
is less reliable. Before replacing a model artifact, the pipeline verifies that
every direct stage has sufficient labels across the recent window. If it does
not, retraining is blocked and the last approved artifact should be used via
prediction-only rather than reintroducing older noisy labels.

Older configs remain available for comparison:

- `v2.1`: legacy Random Forest baseline
- `v2.2`: LightGBM signed-target baseline
- `v2.3`: previous feature-upgrade default
- `v2.4`: current canonical Stage/Phase pipeline with OOF Ridge termination
- `v2.5`: deprecated compatibility configuration

See `docs/models/model_versions.md`.

## Critical Stage/Phase Rules

Raw Redshift `stage_id` is not stable after the 2026 data transition.  Use
`app/services/canonical_events.py`.

Canonical training mapping:

- raw `Stage 1` -> canonical `stage_id = 1`
- raw `Stage 2` -> canonical `stage_id = 2`
- raw `Stage 3` -> canonical `stage_id = 3`
- raw `Phase 2` -> canonical `stage_id = 3` (`Stage 3 / Phase 2`)
- `Phase 1` is canonical `stage_id = 4`, generated from paired Stage 1/2 midpoint labels
- placeholder dates such as `12-31` are excluded

Do not backfill missing Stage 1/2 labels from observed Phase 1.  Phase 1 is a
direct model target only through paired Stage 1/2 midpoint pseudo-labels.

## Manual Curated Events

`app/services/manual_events.py` contains 11 clear single-date mentor-provided
observations from:

`notebooks/experiments/manual_stage_data_validation_report.ipynb`

These rows are appended to the processed Redshift/canonical events at runtime
when enabled in `v2.4`:

```yaml
manual_events:
  enabled: true
  include_in_training: true
  include_in_validation: true
```

They do not write to Redshift.  They become normal labeled events for weather
pulls, feature preparation, training, and validation.  Missing, ambiguous, and
interval manual observations are intentionally not used as direct labels.

Manual rows are filtered by the pipeline `today` cutoff to avoid backtesting
future leakage.

## Architecture

```text
Redshift events
  -> DataPreparationService.pull_weevil_data()
  -> canonical event mapping
  -> optional manual event injection
  -> griddedweather weather pull/cache
  -> process_weather_data()
  -> prepare_training_data()
  -> model_manager.train()
  -> predict + learned termination
  -> Stage/Phase business output mapping
  -> local/S3 outputs
```

## Important Directories

```text
app/
  config/       YAML model configs
  models/       BaseModelManager, RF/LGBM managers, factory
  pipeline/     train_predict, predict_only, backtesting, hindcast, eval, termination
  services/     Redshift, weather/data prep, canonical events, manual events
docs/           design and handoff documents
notebooks/      production and experiment notebooks
outputs/        local generated artifacts
scripts/        operational utilities
tests/          pytest suite
```

## Important Files

| File | Role |
| --- | --- |
| `app/config/weeviltrak_v2.4.yml` | current default config |
| `app/services/canonical_events.py` | Stage/Phase canonicalization and Phase 1 derivation |
| `app/services/manual_events.py` | curated manual observations |
| `app/services/data_preparation_service.py` | Redshift pull, weather processing, feature prep |
| `app/pipeline/train_predict.py` | main train + predict pipeline |
| `app/pipeline/backtesting.py` | walk-forward backtesting |
| `app/pipeline/predict_only.py` | prediction with saved model |
| `app/pipeline/hindcast.py` | historical what-if predictions |
| `app/pipeline/termination.py` | learned termination logic |
| `app/models/model_manager.py` | RF/LGBM model managers |
| `app/models/factory.py` | model manager creation |
| `app/services/database_service.py` | Redshift connection wrapper |

## Environment

Use Poetry with Python 3.11.

```bash
poetry install
poetry env info
```

Required external credentials are normally provided through `.env` or the shell:

```text
S3_BUCKET_NAME=sps-ds-bucket
REDSHIFT_HOST=<host>
REDSHIFT_USER=<user>
REDSHIFT_PASSWORD=<password>
REDSHIFT_DATABASE=<database>
REDSHIFT_PORT=5439
```

Notebook kernel from Poetry:

```bash
poetry run python -m ipykernel install --user \
  --name weeviltrak-poetry \
  --display-name "WeevilTrak Poetry"
```

## Common Commands

```bash
# Train + predict with v2.4
poetry run python -m app.pipeline.train_predict

# Predict only
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
poetry run pytest tests/ -q
```

Quality:

```bash
poetry run black app/ tests/ scripts/
poetry run isort app/ tests/ scripts/
poetry run flake8 app/ tests/
poetry run mypy app/
```

## Coding Conventions

- Instantiate models via `app.models.factory.create_model_manager`.
- Predict through `model_manager.predict(X)`, not the raw estimator.
- Use `PipelineBase` for shared service initialization.
- Keep behavior versioned in YAML configs.
- Do not silently change historical config semantics.
- Do not hardcode credentials.
- Preserve user changes in a dirty worktree.
- Prefer moving notebook logic into tested `app/` modules for production use.

## External Data

- Redshift table: `europe_dna.europe_dna_sps_weeviltrak`
- S3 bucket: `sps-ds-bucket`
- griddedweather cache: `s3://sps-ds-bucket/gridded-weather/cache/10by10/`
- raw WeevilTrak upload area: `s3://sps-ds-bucket/weeviltrak_data/raw_data/`

Tests should not depend on live Redshift or S3 unless explicitly scoped as
integration checks.

## Output Locations

Use local `outputs/` for generated artifacts:

- `outputs/backtesting/<version>/`
- `outputs/hindcast/<version>/`
- `outputs/business_checks/`
- `outputs/validation/`
- `outputs/cache/`
- `outputs/logs/`

Use YAML `output_path` for versioned S3 output paths.

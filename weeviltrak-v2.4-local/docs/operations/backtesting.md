# Historical Backtesting Plan (v2.1/v2.3)

> **Historical document.** This file describes legacy v2.1/v2.3 workflows,
> including expanding-window assumptions and retired paths. Do not use it as
> the v2.4 operating procedure; see
> [`v24_training_release_runbook.md`](v24_training_release_runbook.md).

## 1. Objective

Backtest the **WeevilTrak v2.1 Random Forest model** to measure real 7-day-lead forecast accuracy across multiple years and life stages.

**Core definition:** For each location with a known stage occurrence in 2023/2024/2025, set `testing_date = stage_date − 7 days`, train the model on data strictly before a yearly cutoff, and compare predicted vs actual stage dates.

## 2. Process Overview

The backtesting pipeline has three steps, implemented identically in:
- **`notebooks/backtesting_prod.ipynb`** — interactive / production notebook (canonical source of truth)
- **`app/pipeline/backtesting.py`** — `BacktestingFramework` class for scripted CLI runs

> **Post-reorganization (Phase 2+):** `BacktestingFramework` inherits from `PipelineBase` (`app/pipeline/base.py`) instead of duplicating service initialization. See `docs/repo_reorganization_plan.md`.

### Step 1 — Build Backtesting Plan

From Redshift weevil data, filter to `year ∈ {2023, 2024, 2025}` and `stage_id ∈ {1, 2, 3}`.

```text
testing_date = stage_date − 7 days
```

Build mapping:

```python
date_to_locations: Dict[datetime.date, List[str]]  # testing_date → [location_id, …]
```

Save `backtesting_plan_events.csv` as the audit trail / source of truth.

### Step 2 — Pull and Process Weather Once

Pull weather data from S3 (10×10 km gridded cache) up to the latest `testing_date`, then process features (GDD, rolling/cumulative aggregates) into a single DataFrame:

```python
processed_weather_unique  # ONE row per (date, location_id)
```

**Critical:** Raw weather data contains 2–8 duplicate rows per `(date, location_id)` due to overlapping grid cells. De-duplication (`drop_duplicates(subset=["date", "location_id"], keep="first")`) is required to prevent inflated training sets and duplicate predictions.

> **Post-reorganization (Phase 4):** Weather ETL moves to `app/services/weather_service.py` and feature engineering (GDD, rolling/cumulative) moves to `app/services/feature_engineering.py`. `DataPreparationService` remains the orchestrator but delegates to these extracted services. Method calls from the backtesting loop stay the same — `prepare_training_data()` and `prepare_test_data()` are unchanged.

### Step 3 — Production Backtesting Loop (Train Once Per Year)

Train **one** model per prediction year using an expanding window (cutoff = Jan 1 of that year):

| Predict Year | Cutoff | Training Data |
|-------------|--------|---------------|
| 2023 | 2023-01-01 | 2018–2022 |
| 2024 | 2024-01-01 | 2018–2023 |
| 2025 | 2025-01-01 | 2018–2024 |

For each `testing_date` in that year:

1. **Predict** — build test features for the date, filter to scheduled locations, run `model_manager.predict()` (post-Phase 3) or `model.predict()` (current)
2. **Aggregate** — if duplicate `(prediction_date, location_id, stage_id)` keys exist, collapse using **median** `predicted_days`
3. **Save** — append row to incremental CSV checkpoint (`.tmp.csv`); supports resume after interruption
4. **Log** — record status (`ok`/`skip`/`fail`), reason, and row counts per date

After the loop, convert `.tmp.csv` → `.parquet` for the final deliverable.

**Alternative mode (legacy):** Train a fresh model for every `testing_date` via `predict_for_date()`. Much slower (~N× slower), kept for experimentation. Set `training_window_years=<int>` for a rolling window instead of expanding.

### Step 4 — Evaluate 7-Day Forecast Error

Matching rule:
```
expected_actual_stage_date = prediction_date + 7 days
```
Merge predictions with actuals on `(location_id, stage_id, expected_actual_stage_date == stage_date)`.

```
error_days = predicted_stage_date − actual_stage_date
```
Positive = predicted too late; negative = predicted too early.

Metrics computed per `(prediction_year, stage_id)` and per `prediction_year`:
- `n_predictions`, `mean_error`, `median_error`, `mae_error`
- `within_3d` (fraction ≤ 3 days), `within_5d` (fraction ≤ 5 days)
- `p90_abs_error` (90th percentile of |error|)

## 3. Output Files

All written to a **versioned** output directory derived from the `version` field in the YAML config:
- Notebook/test artifacts: `outputs/test_runs/` or `outputs/backtesting/<version>/`
- CLI: `outputs/backtesting/<version>/` (e.g. `outputs/backtesting/v2.3/`)
- S3: `<output_path>/` which includes the version (e.g. `weeviltrak_data/api-testing/v2.3/`)

This ensures different model versions never overwrite each other's results.

### 3.1 Backtesting Plan

| File | Fields |
|------|--------|
| `backtesting_plan_events.csv` | `location_id`, `stage_id`, `year`, `stage_date`, `testing_date`, `testing_date_only`, `latitude`, `longitude` |

### 3.2 Prediction Results

| File | When Written | Notes |
|------|-------------|-------|
| `predictions_detail_lead7_<ts>.tmp.csv` | Appended after every date | Checkpoint; survives interruption |
| `predictions_detail_lead7_<ts>.parquet` | After loop completes | Converted from `.tmp.csv` |
| `predictions_detail.parquet` | After loop completes | Latest copy (overwritten) |

**Schema:**
```
location_id             str
stage_id                int         (1, 2, or 3)
latitude                float
longitude               float
prediction_date         datetime
prediction_year         int         (year of testing_date)
model_train_cutoff_date str         (ISO date, Jan 1 of prediction_year)
model_train_end_year    int         (prediction_year − 1)
predicted_days          float       (days from prediction_date to predicted stage)
predicted_stage_date    str         (YYYY-MM-DD)
n_rows                  int         (present only when aggregation occurred)
```

### 3.3 Evaluation Outputs

| File | Contents |
|------|---------|
| `evaluation_lead7_<ts>.csv` | One row per matched prediction with `error_days`, `location_state`, `location_name` |
| `testing_results_year_stage_lead7_<ts>.csv` | Metrics per `(prediction_year, stage_id)` |
| `testing_results_year_overall_lead7_<ts>.csv` | Metrics per `prediction_year` |
| `testing_results_year_stage.csv` | Latest copy (overwritten) |
| `testing_results_year_overall.csv` | Latest copy (overwritten) |

### 3.4 Run Logs

| File | Contents |
|------|---------|
| `run_log.csv` | Appended after every date: `prediction_year`, `model_train_cutoff_date`, `testing_date`, `n_locations`, `n_predictions`, `n_predictions_raw`, `aggregated`, `status`, `reason` |
| `run_log_final.csv` | Written at end (same schema) |
| `run_summary.csv` | Aggregated status counts (`ok`/`skip`/`fail` × n_dates) |
| `failures.csv` | Per-date errors (appended mid-run) |
| `failures_final.csv` | Written at end if any failures |

Metrics: `n_predictions`, `mean_error`, `median_error`, `mae_error`, `within_3d`, `within_5d`, `p90_abs_error`

## 4. Key Design Choices

### 4.1 Why Train Once Per Year (Not Once Per Date)?

The original design retrained for every `testing_date` (~99% of runtime was `RF.fit`).
Production mode trains **ONE model per year** — same expanding-window guarantee, ~N× faster.

```
Year 2023: cutoff = 2023-01-01  →  train on 2018–2022
Year 2024: cutoff = 2024-01-01  →  train on 2018–2023
Year 2025: cutoff = 2025-01-01  →  train on 2018–2024
```

The `predict_for_date()` function (trains + predicts per date) is kept for experimentation.

### 4.2 Depend on OOB / CI?

- Current notebook **does not depend on CI / OOB**

- Does not call `ModelManager.predict(..., alpha=...)`

- Can be safely turned off

### 4.3 How to Solve Duplicate Predictions?

- Reason: Multiple rows on the same day in the weather table

- Solution:

  - Remove duplicates at the **weather layer** (de-dup on `(date, location_id)`)

  - `aggregate_predictions_unique()` collapses any remaining duplicates using median `predicted_days`

## 5. How to Run

### Script (recommended for long runs):

```bash
nohup poetry run python -m app.pipeline.backtesting > backtest_nohup.log 2>&1 &
tail -f backtest_nohup.log
```

### Notebook (recommended for production backtesting):

```bash
nohup jupyter nbconvert --to notebook --execute notebooks/backtesting_prod.ipynb \
  --output backtesting_prod.ipynb > backtest_nohup.log 2>&1 &
```

### S3 Upload:

After backtesting completes, results can be uploaded to S3:

```python
s3_manager.upload_file(predictions_df, f"{bucket_name}/weeviltrak_data/api-testing/v2.1/<filename>.csv")
s3_manager.upload_file(eval_df, f"{bucket_name}/weeviltrak_data/api-testing/v2.1/<filename>.csv")
```

Note: The S3 path includes the version from `output_path` in the YAML config. `BacktestingFramework.run_backtest(upload_to_s3=True)` handles this automatically.

## 6. Repo Reorganization Impact

See `docs/repo_reorganization_plan.md` for full details. Summary of phases that affect backtesting:

| Phase | Change | Backtesting Impact |
|-------|--------|--------------------|
| 1 | Delete `app/core/`, `app/api/`, `app/main.py`, `weather_data_collector.py` | No impact — backtesting does not import these |
| 2 | Extract `PipelineBase` to `app/pipeline/base.py` | `BacktestingFramework` inherits `PipelineBase`; `__init__` simplified |
| 3 | Model factory + `BaseModelManager` ABC | Instantiate via `create_model_manager(config, s3_manager)` instead of `ModelManager(config, s3_manager)`; predictions route through `ModelManager.predict()` instead of raw `model.predict()` |
| 4 | Split `DataPreparationService` → `WeatherService` + `FeatureEngineering` | `BacktestingFramework.prepare_weather()` calls same orchestrator API; internal delegation is transparent |
| 5 | Add `HindcastPipeline` | New sibling pipeline; shares `PipelineBase`. Distinction: hindcast loads a pre-trained model and predicts arbitrary past dates without training or evaluation |

### Post-Phase 3: Model Instantiation

```python
# Before (current)
model_manager = ModelManager(config, s3_manager)

# After Phase 3
from app.models.factory import create_model_manager
model_manager = create_model_manager(config, s3_manager)  # reads model_type from YAML
```

### Post-Phase 3: Prediction Routing

```python
# Before (current) — bypasses ModelManager, calls raw sklearn
y_pred = model.predict(x_test[config.features])

# After Phase 3 — routes through ModelManager.predict()
y_pred = model_manager.predict(x_test[config.features])
```

### Post-Phase 4: YAML Config Additions

```yaml
# weeviltrak_v2.1.yml — new fields
version: v2.1              # used for versioned output directories (local + S3)
model_type: random_forest   # used by factory; default for v2.1
output_path: weeviltrak_data/api-testing/v2.1  # S3 prefix includes version
```

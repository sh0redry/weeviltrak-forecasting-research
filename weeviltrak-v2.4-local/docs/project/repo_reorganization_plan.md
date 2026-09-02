# WeevilTrak Repo Reorganization Plan

**Date:** 2025-03-21  
**Goal:** Reorganize the repo to separate the ML pipeline from unused FastAPI scaffolding, introduce a model factory pattern (supporting future LightGBM+GP), extract a shared pipeline base to remove duplication, break up the god-class `DataPreparationService`, and add a **hindcast** pipeline for predicting past events for research.

Each phase is independently verifiable and doesn't break existing notebook workflows.

---

## Phase 0: Set Up Test Infrastructure

0. **Create `tests/` directory with pytest scaffolding** — Add `tests/__init__.py`, `tests/conftest.py` with shared fixtures (mock config, mock S3, mock database connections). Ensure pytest is configured in `pyproject.toml` (already set to `testpaths = ["tests"]`).

**Verification:** `pytest` runs and prints "no tests collected" (0 tests is fine; infrastructure is ready).

---

## Phase 1: Clean Up Dead Code & Consolidate Logging

1. **Remove FastAPI scaffold** — Delete `app/main.py`, `app/core/` (config.py and logging.py), and `app/api/`. The FastAPI layer is broken (non-existent `app/api/endpoints/` causes import failures), adds no value to the ML pipeline, and its scaffolding is unused. If HTTP endpoints are needed in the future, re-scaffold then.
2. **Remove dead `WeatherDataCollector`** — `app/services/weather_data_collector.py` is never imported by the pipeline; its logic is duplicated inside `DataPreparationService`.
3. **Consolidate logging to `app/settings.py`** — Remove `app/core/logging.py`. Use the existing `setup_logger()` from `app/settings.py` as the single logging system throughout the pipeline.
4. **Prune FastAPI-stack dependencies from `pyproject.toml`** — Remove: fastapi, uvicorn, pydantic-settings, sqlalchemy, alembic, psycopg2-binary, python-jose, passlib, python-multipart, httpx, celery, redis. These are unused and bloat the install. Retain pydantic core for general data validation if needed elsewhere.

**Verification:** `poetry install` succeeds; all notebooks and pipeline scripts import cleanly and use unified logging.

---

## Phase 2: Extract Shared Pipeline Base

5. **Create `app/pipeline/base.py`** — Extract the identical 4-service initialization (config, DB, S3, data service, model manager) that's copy-pasted between `WeevilTrakPipeline.__init__()` and `BacktestingFramework.__init__()` into a `PipelineBase` class. Both classes inherit from it.
6. **Refactor `model_eval.py`** — Accept services from outside rather than creating its own config/S3 instances internally.

**Verification:** `backtesting_prod.ipynb` and `test_train_predict.ipynb` produce identical results.

---

## Phase 3: Model Factory Pattern (Future Model Support)

7. **Define model interface** — New `app/models/base.py` with `BaseModelManager` ABC defining:
   - `train(X, y) -> None` — fit the model
   - `predict(X) -> np.ndarray` — returns 1D array of predictions (matching sklearn's `RandomForestRegressor.predict()` signature)
   - `save_model(path: str) -> None` — serialize to S3 or local
   - `load_model(path: str) -> None` — deserialize from S3 or local

   All future models (LightGBM, GP, etc.) must implement `predict()` with identical signature/return type as `RandomForestRegressor` so that downstream pipeline code (inference loops, confidence interval computation) works uniformly.

8. **Update `ModelManager` to implement `BaseModelManager`** — Add `BaseModelManager` as parent class. Fix the pipeline to route predictions through `ModelManager.predict()` instead of bypassing it with raw `model.predict()` — update all 3 call sites in `train_predict.py` and `backtesting.py`.

9. **Create model factory** — New `app/models/factory.py` with `create_model_manager(config, s3_manager)` that reads `model_type` from YAML config. Instantiates the underlying model (RandomForest for v2.1, LightGBM+GP for v2.2) and wraps it in `ModelManager`.

10. **Add `model_type: random_forest` and `version: v2.1` to YAML** — Backward-compatible defaults in `app/config/weeviltrak_v2.1.yml`. The `version` field is used to create versioned output directories (local and S3) so different model versions never overwrite each other's results.

**Verification:** Factory returns `ModelManager` wrapping RandomForest by default, prediction outputs identical.

---

## Phase 4: Break Up DataPreparationService

11. **Extract weather ETL → `app/services/weather_service.py`** — Move `query_weather_data_for_weeviltrak_locs()`, `_remap_location_id()`, `_create_weather_config()`, `_normalize_weather_columns()`. This service will be responsible for:
    - Reading weather from S3 cache at `s3://sps-ds-bucket/gridded-weather/cache/10by10/` (default griddedweather cache location)
    - Falling back to Redshift pulls if cache is missing for a date range
    - Returning normalized weather DataFrames ready for feature engineering
12. **Extract feature engineering → `app/services/feature_engineering.py`** — Move `process_weather_data()` and all private helpers (`_convert_celsius_to_fahrenheit`, `_calc_growing_degree_days`, `_calc_annual_cumulative`, `_get_pest_year`, `_calc_rolling_avg`).
13. **Keep `DataPreparationService` as orchestrator** — Retains `pull_weevil_data()`, `prepare_training_data()`, `prepare_test_data()`, delegates weather/feature work to extracted modules. Shrinks from ~870 → ~300 lines.
14. **Unify `prepare_test_data()` and `prepare_test_data_latest()`** — Nearly identical methods that differ only in keeping/dropping `location_id`. Merge into one with a `keep_location_id` parameter.

**Verification:** Unit tests for feature engineering (GDD, cumulative sums, rolling averages). Backtest outputs match pre-refactor.

---

## Phase 5: Add Historical Prediction (Hindcast) Pipeline

15. **Create `app/pipeline/hindcast.py`** — New `HindcastPipeline(PipelineBase)` for the current frozen `v2.2` LightGBM model:

    ```python
    def run_hindcast(
      self,
      target_dates: List[str],    # arbitrary past dates to predict for
      stages: List[int] = [1, 2, 3],
      location_ids: List[str] | None = None,
      locations: List[dict] | None = None,  # {"name": "...", "latitude": ..., "longitude": ...}
      output_dir: str = None,     # None → derives versioned dir from config (e.g. hindcast_results/v2.2/)
      join_actuals: bool = False   # optionally join to real weevil observations
    ) -> pd.DataFrame
    ```

    | Pipeline | Training | Prediction Dates | Evaluation | Use Case |
    |---|---|---|---|---|
    | `WeevilTrakPipeline.run_prediction()` | Loads pre-trained | Single future date | No | Operational |
    | `BacktestingFramework.run_backtest()` | Trains per-year | All dates in year range | Yes (MAE, etc.) | Model validation |
    | **`HindcastPipeline.run_hindcast()`** | **Loads pre-trained** | **Arbitrary past dates** | **Optional join only** | **Research / what-if** |

    - Uses a **frozen pre-trained v2.2 model** by default (`hindcast_model_path`) — no training
    - Frozen research baseline is trained through **2025-12-31 inclusive**
    - Supports both existing `location_id` inputs and custom latitude/longitude inputs
    - For custom coordinates, reuses the nearest-centroid mapping logic from `tests/predict_specific_locations.ipynb`
    - Pulls weather for historical dates via weather service (Phase 4 extraction):
      - **Default weather cache:** `s3://sps-ds-bucket/gridded-weather/cache/10by10/` (griddedweather default)
      - **Frozen model path:** Defined in the model config file (e.g., `weeviltrak_v2.2.yml` via `hindcast_model_path`)
      - Avoids Redshift re-pulls whenever cache objects exist for requested date range
    - Outputs raw predictions: `prediction_date`, `location_id`, `place_id`, `latitude`, `longitude`, `stage_id`, `raw_signed_days`, `predicted_days`, `predicted_stage_date`
    - Optional `join_actuals=True` adds: `actual_stage_date`, `error_days` using the next actual stage date on or after `prediction_date`
    - Deliberately simpler than backtesting — no training loop, no evaluation metrics, no aggregation

16. **Create `notebooks/hindcast.ipynb`** — Demo notebook: load model, select dates/regions, run hindcast, visualize predictions vs actuals on Google Maps.

**Verification:** Hindcast for 5 known 2023 dates matches backtesting results for same dates when using the same model; weather is served from S3 cache (no Redshift hit) unless missing.

---

## Phase 6: Final Directory Structure

```
app/
  __init__.py
  settings.py                          # Single unified logging system
  config/
    weevilltrak_config.py
    weeviltrak_v2.1.yml                # RF config (version: v2.1, model_type: random_forest)
  models/
    base.py                            # BaseModelManager ABC — NEW
    factory.py                         # create_model_manager() — NEW
    model_manager.py                   # ModelManager (implements BaseModelManager, predict() route fixed)
  pipeline/
    base.py                            # PipelineBase (shared init) — NEW
    train_predict.py                   # WeevilTrakPipeline(PipelineBase)
    backtesting.py                     # BacktestingFramework(PipelineBase)
    hindcast.py                        # HindcastPipeline(PipelineBase) — NEW
    predict_only.py                    # CLI wrapper
    model_eval.py                      # Evaluation functions
  services/
    data_preparation_service.py        # Orchestrator (~300 lines, down from 900)
    database_service.py                # Redshift connection
    weather_service.py                 # Weather ETL — NEW (extracted from DataPreparationService)
    feature_engineering.py             # Feature engineering — NEW (extracted from DataPreparationService)
data/
docs/
logs/
notebooks/
  hindcast.ipynb                       # Demo notebook for hindcast exploration
  backtesting_prod.ipynb
  ...
scripts/
tests/
  __init__.py                          # NEW
  conftest.py                          # NEW (shared fixtures)
```

**Removed:**
- `app/main.py` (FastAPI entry)
- `app/core/` (FastAPI config and logging)
- `app/api/` (broken endpoints)
- `app/services/weather_data_collector.py` (dead code)

**Added:**
- 5 new files (`base.py`, `factory.py`, `hindcast.py`, `weather_service.py`, `feature_engineering.py`)
- 1 new notebook (`hindcast.ipynb`)
- Test scaffolding (`tests/__init__.py`, `tests/conftest.py`)

---

## Architectural Issues Addressed

| # | Issue | Severity | Phase |
|---|-------|----------|-------|
| 1 | Two separate apps in one repo (dormant FastAPI + ML pipeline) | High | 1 |
| 2 | Dead API endpoints (`app/api/endpoints/` doesn't exist) | High | 1 |
| 3 | Duplicate service initialization in pipeline classes | High | 2 |
| 4 | Two logging systems (`settings.py` vs `core/logging.py`) | Medium | 1 |
| 5 | Dead code (`WeatherDataCollector` never used) | Medium | 1 |
| 6 | God class (`DataPreparationService` ~900 lines) | Medium | 4 |
| 7 | `ModelManager.predict()` bypassed by pipeline code | Medium | 3 |
| 8 | No model abstraction for future model types | Medium | 3 |
| 9 | Hardcoded data-quality filters in code | Low | 4 |
| 10 | Unused FastAPI-stack dependencies bloating install | Low | 1 |
| 11 | No test infrastructure (zero test coverage) | Medium | 0 |

---

## Decisions

- **FastAPI layer**: Remove entirely (Phase 1). It's broken scaffolding (imports crash due to nonexistent `app/api/endpoints/`), adds no value to the ML pipeline, and its deps bloat the install. If HTTP endpoints are needed in the future, re-scaffold from scratch at that time.
- **Phased execution**: Phase 0 (test infrastructure) → Phase 1 (cleanup) → Phase 2 (base) → Phases 3+4 parallel (factory + service decomposition) → Phase 5 (hindcast).
- **Hindcast vs backtesting**: Hindcast is deliberately simpler — no training, no evaluation metrics. It's a "what would the model have predicted?" research tool.

---

## Further Considerations

0. **Notebook retention policy**: Notebooks are for research/exploration only. After Phase refactoring, audit all notebooks:
   - `backtesting_prod.ipynb`, `test_train_predict.ipynb`: Used for Phase 2 verification. Keep only if they document model validation workflow that can't be automated as scripts/tests.
   - `hindcast.ipynb`: Created in Phase 5 as demo. Keep only if it provides researcher-friendly exploration that scripts don't cover.
   - Delete any notebook that is outdated, duplicative, or unused. Better to have no notebook than stale documentation.

1. **Cache configuration strategy**:
   - **Weather cache (griddedweather):** Always reads from default S3 location `s3://sps-ds-bucket/gridded-weather/cache/10by10/`
   - **Model output cache:** Config-driven — location specified in model config file (e.g., `weeviltrak_v2.1.yml`). Each model version can define its own cache path.
   - This allows developers to cache model predictions separately per model version without conflicts.

2. **Tests for Phase 4 feature engineering**: Phase 0 sets up pytest infrastructure. After Phase 4's extraction, add unit tests for pure functions: GDD calculation, cumulative aggregates, rolling averages, `remap_location_id()` behavior.

3. **Config versioning**: When LightGBM+GP arrives, create a separate `weeviltrak_v2.2.yml` alongside v2.1 (rather than making `model_type` a runtime toggle). Each model architecture gets its own config file. The `version` field in each YAML (e.g. `v2.1`, `v2.2`) drives versioned output directories:
   - Local: `backtesting_results/<version>/`, `hindcast_results/<version>/`
   - S3: `weeviltrak_data/api-testing/<version>/`
   - This prevents different model versions from overwriting each other's results.

4. **CLI entry point for backtesting**: Currently lives only in notebooks. Adding `python -m app.pipeline.backtesting --years 2022 2023 ...` would improve reproducibility while keeping the notebook for interactive exploration.

5. **Minimal tests cover config features**: Test that the config-defined feature set (GDD, cumulative/rolling aggregates for `cumu_gdd_air`, `rolling_gdd_air`, `rolling_humidity_mean_pct`, `cumu_precip_total_mm`) are computed correctly in feature engineering.

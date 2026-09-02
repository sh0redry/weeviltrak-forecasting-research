# v2.3 Issue 2 Full Feature And Validation Plan

## 人话版：这次要解决什么问题

这次我们要解决的，已经不再是“事件发生以后为什么模型还不给 0”这个问题本身了。那部分在 `v2.2` 里已经通过 signed target、post-event supervision 和 `cumu_gdd_air` 的单调约束解决了一大块。

`v2.3` 要处理的是另一个更偏 pre-event 的问题，也就是 **Issue 2 的 cold snap sensitivity**。

用人话说，就是：

- Stage 1 在春季推进过程中，原来的特征对短期冷暖波动太敏感
- 3 月上旬到中旬如果先暖后冷，预测会出现明显反弹
- 一周左右的寒潮会把 `PredictedDays` 往后推很多天
- 模型缺少一个稳定的季节锚点，也缺少对“冷应激”本身的直接表达

所以这次不是要换掉当前模型路线，而是要在 `v2.2` 的基础上，新增一个 **`v2.3` 特征增强版本**，通过更稳定的特征来改善这个问题。

## Human Summary

Issue 2 is now understood primarily as a **pre-event stability problem**, not a post-event zero-state problem. The current `v2.2` path already improved Issue 1 by introducing:

- `signed_days_to_event`
- post-event training supervision
- a LightGBM monotonic constraint on `cumu_gdd_air`

Those changes helped the model behave better near and after the event date, but they do not fully address the remaining warm-then-cold reversal behavior seen in Stage 1 during early spring.

The `v2.3` goal is therefore:

- keep the current LightGBM model family
- preserve the existing signed-target and post-event behavior
- improve feature stability by adding:
  - `14-day rolling_gdd_air`
  - `doy`
  - `cdd_air`
  - `cumu_cdd_air`
  - `rolling_cdd_air`

This is a feature-and-validation upgrade, not a model-family replacement.

## Final Design Decisions

- Create a **new** model version `v2.3`; do not modify `v2.2`
- Keep `target = signed_days_to_event`
- Keep business output semantics unchanged:
  - `predicted_days >= 0`
  - `predicted_stage_date` is computed from clipped business output
- Keep internal diagnostics compatible with `raw_signed_days`
- Add `rolling_window_days: 14` in the new config
- Add `doy` using the repo’s existing historical definition:
  - `date.dt.dayofyear`
- Keep `gdd_air` logic unchanged
- Add `cdd_air`, `cumu_cdd_air`, and `rolling_cdd_air`
- Keep LightGBM monotonic constraints only on:
  - `cumu_gdd_air: -1`
- Do not add monotonic constraints to `CDD` features in this phase
- Do not introduce a new pipeline branch; reuse current `config.features` plumbing

## Versioning And Public Behavior

### Versioning rule

`v2.2` remains frozen as the Issue 1 baseline.

`v2.3` will be a new config and output/model path intended to test whether feature improvements reduce Issue 2 sensitivity without altering the existing `v2.2` behavior.

### Public behavior

The public business-facing output remains unchanged:

- `predicted_days` must remain clipped to `>= 0`
- `predicted_stage_date` remains the business event date output

Internal diagnostic output may still include:

- `raw_signed_days`

This allows the repo to preserve the current signed-target training semantics while keeping business outputs backward-compatible.

## Configuration Changes

### New config file

Create a new configuration:

- `app/config/weeviltrak_v2.3.yml`

This file should be based on `v2.2` but must be independent from it.

### Required config fields

`v2.3` should include:

- `target: signed_days_to_event`
- `rolling_window_days: 14`
- `features` containing:
  - `stage_id`
  - `cumu_gdd_air`
  - `rolling_gdd_air`
  - `rolling_humidity_mean_pct`
  - `cumu_precip_total_mm`
  - `doy`
  - `cumu_cdd_air`
  - `rolling_cdd_air`
  - `latitude`
  - `longitude`

### Monotonic constraints

Preserve the current monotonic constraint scope:

- `cumu_gdd_air: -1`

Explicit non-goal for this phase:

- do not add monotonic constraints to `cdd_air`
- do not add monotonic constraints to `cumu_cdd_air`
- do not add monotonic constraints to `rolling_cdd_air`

### Paths

Assign new `model_path` and `output_path` for `v2.3`.

Explicit rule:

- do not overwrite the paths used by `app/config/weeviltrak_v2.2.yml`

## Feature Engineering Design

Primary implementation area:

- `app/services/data_preparation_service.py`

### Existing logic that stays unchanged

The following should remain unchanged:

- target construction logic
- signed-target behavior
- post-event augmentation behavior
- `gdd_air` formula and base temperature
- current prediction output clipping semantics

Older configs must still run without being forced to use the new features.

### `doy`

Definition:

- `doy = date.dt.dayofyear`

Reason:

- this matches the repo’s historical `day_of_year` definition already used in `v2.1`
- it keeps `v2.3` aligned with the existing interpretation of “day of year”

### `gdd_air`

Definition remains unchanged:

- use the existing base-50F growing degree day calculation

There is no semantic change to `gdd_air` in this phase.

### `rolling_gdd_air`

Behavior:

- continue grouped rolling within `(pest_year, place_id)`
- read window length from config where available
- `v2.3` will use `14`

Implementation expectation:

- older configs may fall back to the current default behavior
- `v2.3` must explicitly receive a 14-day window through configuration

### `cdd_air`

Definition:

```python
mean_temp_f = (air_temp_min_f + air_temp_max_f) / 2
cdd_air = max(0, 50 - mean_temp_f)
```

Business meaning:

- `cdd_air` encodes cold intensity below the same 50F base used for GDD
- this gives the model a direct signal for cold-stress periods that are currently invisible once GDD clips to zero

### `cumu_cdd_air`

Definition:

- cumulative sum of `cdd_air` within `(pest_year, place_id)`

This is the season-level cold-load feature.

### `rolling_cdd_air`

Definition:

- grouped rolling mean of `cdd_air` using the same configured rolling window

For `v2.3`:

- `rolling_cdd_air` uses a 14-day window

## Pipeline Impact

No new pipeline branch is needed.

Affected paths:

- training path in `app/pipeline/train_predict.py`
- backtesting path in `app/pipeline/backtesting.py`
- prediction path that uses processed weather and `config.features`

Expected behavior:

- once `process_weather_data()` produces the new columns
- and once `v2.3` requests them in `config.features`
- existing training, prediction, and backtesting code will automatically use them

This plan intentionally reuses current plumbing rather than introducing a dedicated `v2.3` execution path.

## Codex Execution Plan

### A. Create the new config

Primary file:

- `app/config/weeviltrak_v2.3.yml`

Implementation intent:

- copy the `v2.2` LightGBM structure
- preserve signed-target and post-event fields
- add:
  - `rolling_window_days: 14`
  - `doy`
  - `cumu_cdd_air`
  - `rolling_cdd_air`
- assign new model/output paths

### B. Extend weather feature engineering

Primary file:

- `app/services/data_preparation_service.py`

Implementation intent:

- compute `doy`
- compute `cdd_air`
- include `cdd_air` in cumulative features
- include `cdd_air` in rolling features
- make rolling window configurable from the model config

Recommended implementation shape:

1. Keep temperature conversion as-is.
2. Compute `gdd_air` using the current logic.
3. Compute `doy` directly from `date`.
4. Compute `cdd_air` from the Fahrenheit mean temperature.
5. Extend cumulative feature generation to include `cdd_air`.
6. Extend rolling feature generation to include `cdd_air`.
7. Read rolling window size from config where available.
8. Preserve compatibility for older configs that do not request new features.

### C. Preserve target and output behavior

Primary areas:

- `app/services/data_preparation_service.py`
- `app/pipeline/train_predict.py`
- `app/pipeline/backtesting.py`

Implementation intent:

- do not change signed-target behavior
- do not change post-event augmentation behavior
- do not change business output clipping semantics

This phase is only a feature-engineering and validation upgrade.

## Validation Strategy

Validation is divided into three layers.

### Layer 1: Implementation Correctness

This layer is mandatory.

It answers:

- did we actually compute the new features correctly?
- did the new config wire them into the model path correctly?

#### Required unit-test coverage

Add or update tests covering:

- `doy == date.dt.dayofyear`
- `cdd_air` calculation on warm and cold examples
- `cumu_cdd_air` accumulation by `(pest_year, place_id)`
- `rolling_gdd_air` uses a 14-day window when configured
- `rolling_cdd_air` uses the same configured window
- `v2.3` config contains:
  - `signed_days_to_event`
  - `rolling_window_days: 14`
  - `doy`
  - `cumu_cdd_air`
  - `rolling_cdd_air`
- LightGBM still trains and predicts with the expanded `v2.3` feature set
- existing monotonic-constraint behavior on `cumu_gdd_air` still holds

#### Existing tests that must remain green

- `tests/unit/services/test_post_event_zero_training.py`
- `tests/unit/pipeline/test_signed_target_outputs.py`
- `tests/unit/models/test_lightgbm_monotonic_constraints.py`

Layer 1 is the minimum correctness gate. If it fails, later validation is not trustworthy.

### Layer 2: Issue 2 Targeted Validation

This layer is mandatory.

It answers:

- did `v2.3` actually reduce the warm-then-cold reversal behavior that defines Issue 2?

#### Reference baseline

Use the historical Issue 2 diagnostic figure as the comparison baseline:

- `data/plots/issue2_forecast_drift_reversal.png`

The goal is not to create a new generic leaderboard. The goal is to reproduce the same style of behavioral view and compare whether `v2.3` behaves better.

#### Recommended validation scope

- Stage 1 only
- prediction window: `2026-03-01` through `2026-03-24`
- US-only filtering
- latitude filter aligned to the original issue view
- aggregate by latitude band using the same banding logic as the original issue figure

#### Required outputs

Produce:

- a new `v2.3` drift-reversal figure using the same plotting logic as the original Issue 2 figure
- a detailed CSV with per-date, per-place predictions
- a latitude-band summary CSV or markdown
- a summary markdown comparing old vs new directional behavior

#### Required quantitative summaries

In addition to the figure, summarize:

- per latitude band, maximum upward swing during `2026-03-07` to `2026-03-14`
- per latitude band, net drift from `2026-03-01` to `2026-03-24`
- per latitude band, count of positive jump days in the full window

#### Acceptance direction for Layer 2

`v2.3` should show:

- smaller upward reversals during the cold snap window
- reduced end-of-window drift inflation in affected latitude bands
- no obvious pathological flattening such as widespread premature zeroing

This layer is the issue-specific success check and carries more weight than generic backtesting metrics for this feature.

### Layer 3: General Model Validation

This layer is recommended, but secondary.

It answers:

- did the model become broadly worse outside the targeted Issue 2 scenario?

Use existing backtesting infrastructure to check for large regressions.

Minimum expectation:

- no degradation severe enough to make the model unusable outside the Issue 2 window
- Stage 1 remains the primary interpreted target for this phase

Important interpretation rule:

- Layer 3 is supportive validation
- Layer 2 is the main success criterion for this work

## Validation Deliverables

Write validation artifacts under:

- `docs/implemention_results/`

Recommended outputs:

- `docs/implemention_results/v2_3_issue2_drift_reversal.png`
- `docs/implemention_results/v2_3_issue2_detail.csv`
- `docs/implemention_results/v2_3_issue2_lat_band_summary.csv`
- `docs/implemention_results/v2_3_issue2_validation_summary.md`

If a dedicated validation script is added, it should write exactly these outputs.

## Acceptance Criteria

The work is complete when:

- `v2.3` config exists and is independent from `v2.2`
- weather feature engineering produces:
  - `doy`
  - `cdd_air`
  - `cumu_cdd_air`
  - `rolling_cdd_air`
- `rolling_gdd_air` and `rolling_cdd_air` use a 14-day configured window for `v2.3`
- all Layer 1 tests pass
- Layer 2 validation artifacts are produced
- Layer 2 comparison shows directional improvement versus the original Issue 2 drift figure
- no breaking regression is introduced into signed-target or monotonic-constraint behavior

## Assumptions

- this spec is intentionally written under `docs/implemention/` to match the repo’s existing folder naming
- `doy` must follow the repo’s historical calendar-DOY definition from `v2.1`
- `v2.2` remains unchanged and serves as the Issue 1 baseline
- RF/LGBM aligned-feature business-check redesign is out of scope for this `v2.3` phase

# WeevilTrak Pipeline Handoff Notes

## Purpose

This document explains the previous `v2.2` pipeline behavior after the Stage 1
signed-target update. `v2.3` is now the current default; keep this document as
historical context for the signed-target transition.

It focuses on:

- what changed
- why the change was made
- which files were updated
- what the `v2.2` behavior was
- what remains backward-compatible
- how the change was validated

## Previous `v2.2` Default

The `v2.2` pipeline configuration was:

- config: `app/config/weeviltrak_v2.2.yml`
- model type: `lightgbm`
- training target: `signed_days_to_event`

This means the default pipeline now trains a LightGBM model on a signed target
instead of the legacy Random Forest model on `days_to_event`.

## Why This Changed

The main reason for this update was the Stage 1 event-boundary behavior.

Under the previous default setup, it was too easy to interpret the model as
"not converging" near the true event date. After the signed-target work and the
event-window validation update, the current interpretation is more precise:

- the model can predict signed distance to the event during training
- prediction output keeps the raw signed value for analysis
- business-facing output is still clipped to non-negative remaining days

This separation makes it easier to inspect model behavior near the event date
without changing the external meaning of the production-facing countdown field.

## Prediction Output Semantics

The current prediction flow produces two related fields:

- `raw_signed_days`
- `predicted_days`

Their meanings are:

- `raw_signed_days`: the direct model output from the signed target
- `predicted_days`: the business-facing countdown value, defined as
  `max(raw_signed_days, 0)`

In other words:

- negative raw predictions are preserved for analysis
- business output still reports "days remaining" and therefore never goes below
  zero

`predicted_stage_date` is derived from `predicted_days`, not from
`raw_signed_days`.

## Files Updated In This Change

The following pipeline files were updated so that the default entry points now
use the current configuration:

- `app/pipeline/train_predict.py`
- `app/pipeline/backtesting.py`
- `app/pipeline/predict_only.py`
- `app/pipeline/model_eval.py`

### What Changed In Each File

`app/pipeline/train_predict.py`

- updated the default `__main__` entry point from `weeviltrak_v2.1.yml` to
  `weeviltrak_v2.2.yml`
- clarified in the module header that the default production pipeline uses
  LightGBM with `signed_days_to_event`
- clarified that prediction output now includes both `raw_signed_days` and
  clipped `predicted_days`

`app/pipeline/backtesting.py`

- updated the default `__main__` entry point from `weeviltrak_v2.1.yml` to
  `weeviltrak_v2.2.yml`
- updated the constructor documentation so the default example points to the
  current LightGBM signed-target pipeline
- added a general event-window validation path that uses the same event scope as
  regular backtesting, but evaluates predictions around the actual stage date

`app/pipeline/predict_only.py`

- updated the example command so it points to `weeviltrak_v2.2.yml`

`app/pipeline/model_eval.py`

- updated the default configuration path from `weeviltrak_v2.1.yml` to
  `weeviltrak_v2.2.yml`

## What Did Not Change

This update did not remove backward compatibility with older configurations.

The repository still supports:

- `app/config/weeviltrak_v2.1.yml`
- `days_to_event`
- `random_forest`

This is intentional. Older configurations may still be needed for historical
comparison, regression checks, or explicit RF vs LightGBM analysis.

One important example is:

- `app/pipeline/monotonic_business_check.py`

That file intentionally keeps both `v2.1` and `v2.2` because it is designed as
an A/B comparison tool rather than a single-version production entry point.

## Data Preparation Status

The data preparation layer already supports the current signed-target workflow.

Specifically:

- training data construction includes `signed_days_to_event`
- the legacy `days_to_event` column is still retained for compatibility
- post-event training rows are still added
- prediction output in the pipeline keeps both raw signed output and clipped
  business output

This means the main code path is already aligned with the current modeling
approach. The pipeline changes in this update were primarily about making the
default entry points consistent with that behavior.

## Backtesting Validation Modes

`BacktestingFramework` now supports two validation modes. They answer different
questions and should not be treated as interchangeable.

### Fixed-Lead Backtesting

This is the existing production-style backtesting path:

- method: `run_backtest(...)`
- evaluation: `evaluate_7day_forecast_error(...)`
- prediction date: `testing_date = stage_date - lead_days`
- default lead time: `7` days
- default stages: `1`, `2`, and `3`
- model training cutoff: January 1 of the prediction year

This mode answers the forecast-accuracy question:

- if the model is run `N` days before the actual event, how close is the
  predicted stage date?

The main output is date-error based:

- `error_days = predicted_stage_date - actual_stage_date`
- `mae_error`
- `within_3d`
- `within_5d`
- `p90_abs_error`

Use this mode when the question is about operational forecast quality at a fixed
lead time.

### Event-Window Validation

This is the signed-target behavior check:

- method: `run_event_window_validation(...)`
- prediction dates: `stage_date - days_before` through `stage_date + days_after`
- default window: `[-3, +3]`
- default stages: `1`, `2`, and `3`
- model training cutoff: January 1 of the event year

This mode uses the same event scope as regular backtesting. The selected years,
stages, locations, and actual `stage_date` values come from the normal
backtesting plan. The difference is that each event is expanded into multiple
prediction dates around the actual event date.

This mode answers the event-boundary question:

- does the signed-target model reach zero near the actual event date?
- which stage/location/year windows still fail to terminate?
- which failures are near-zero cases versus severe outliers?

The main output is window-based:

- `terminated`: `predicted_days <= 0`
- `near_zero`: `predicted_days <= zero_day_termination_threshold`
- `termination_rate`
- `near_zero_rate`
- `median_min_predicted_days`
- `median_day0_predicted_days`
- `event_window_non_terminated_windows.csv`

For `v2.2`, this mode is especially useful because `predicted_days` is clipped
from `raw_signed_days`. A zero `predicted_days` value means the raw signed model
output has crossed the event boundary or landed exactly on it.

### Recommended Interpretation

Use both modes when validating a model update.

Fixed-lead backtesting tells us whether the model is useful at a production lead
time. Event-window validation tells us whether the signed-target model behaves
properly near the true event boundary.

A model can look acceptable in one mode and still need review in the other. For
example:

- fixed-lead error can be reasonable while event-boundary termination remains
  unstable
- event-window termination can improve while fixed-lead bias remains too early
  or too late

## Stage 1 Event-Window Validation Result

The Stage 1 `[-3, +3]` event-window validation was run before the generalized
three-stage validation path was added to `BacktestingFramework`.

This result should be read as the current Stage 1 signed-target finding:

- script: `scripts/run_issue1_signed_target_validation.py`
- config: `app/config/weeviltrak_v2.2.yml`
- stage: `1`
- window: `[-3, +3]` around the actual `stage_date`
- validation scope: historical locations and actual Stage 1 event windows

With the `[-3, +3]` window, the model usually gets close to the event and often
reaches zero, but a few severe outliers still remain.

Issue 1, as a question of whether the model can terminate near the event in most
windows, is now largely resolved. The remaining uncovered cases are no longer
mainly about a general failure of the termination mechanism. They are a small
number of more specific problems, including:

- limited training coverage for extremely early events
- unstable generalization at a few individual locations
- weak feature representation in some low-signal cases
- boundary cases that got close to zero but were not forced to zero by
  post-processing

### Overall Result

The Stage 1 validation produced:

- event windows: `80`
- predictions: `560`
- locations: `36`
- windows where the model reached `predicted_days == 0` at least once within
  `[-3, +3]`: `66 / 80` (`82.5%`)
- windows where this happened in the core days `{-1, 0, +1}`: `57 / 80`
  (`71.3%`)
- median absolute error across the full window: `2` days
- median absolute error across the core days: `1` day

On average, `predicted_days` dropped from `2.71` at day `-3` to `1.01` at day
`+3`, so the overall event-window trend looks reasonable.

The error distribution still has a long tail. The maximum error was `28` days,
so performance is not stable for every location.

This means two things are true at the same time:

- the model is now much closer to being usable, because it usually gets near the
  true event and can converge to zero
- the model is still not fully robust, because a small number of windows are
  consistently predicted too late

### Year-Level Interpretation

`2023`

- performance was moderate
- the model could often reach zero
- some windows still stayed above zero near the event
- there were outliers, but most were around `5-7` days rather than the most
  extreme failures

`2024`

- median performance looked acceptable
- a few extreme failures made the year look much worse
- the clearest cases were `location_id` `103` and `104`
- their true Stage 1 date was `2024-03-14`
- the model kept predicting dates around `2024-04-05` to `2024-04-11`
- errors were about `22-28` days

Historically, Stage 1 at these locations usually happened in early April to
early May. The model appears to have fallen back toward the local historical
timing instead of adapting to the unusually early `2024-03-14` event.

So 2024 was not bad everywhere. It was mostly hurt by a few extreme early-event
failures.

`2025`

- this was the best year
- almost all windows reached zero near the event
- predictions were much tighter around the true event date
- from a business perspective, 2025 was closest to the target behavior of
  reliably ending near the event

### Direct Conclusion For Issue 1

Under the new window, Issue 1 is no longer a general "never reaches zero"
problem.

The main remaining problem is failure on a small number of extremely early
events. The highest-priority cases are:

- `location_id` `103` in 2024
- `location_id` `104` in 2024

### Location 103 And 104 In 2024

The true Stage 1 happened on `2024-03-14`.

Across the full `[-3, +3]` window, the model kept predicting that the event was
still `20-30` days away:

- `location_id` `103`: error was about `25-28` days
- `location_id` `104`: error was about `22-26` days

This was not a one-day spike. The whole prediction path was consistently too
late.

The most likely reason is that `2024-03-14` was unusually early for these two
locations compared with their own history. When training the 2024 model, the
only historical samples available for these locations were:

- `2021-04-05`
- `2022-05-02`
- `2023-04-14`

So the model had never seen such an early local event as `2024-03-14`.

At the same time, the event-day features did not look like an obviously
abnormal-input case:

- `cumu_gdd_air` was already around `63-69`
- the latitude was low, so these locations would normally be expected to turn
  earlier than northern sites

The current interpretation is that the model pulled predictions back toward the
usual historical timing for those locations. This looks like a normal failure
mode caused by limited training data rather than an implementation bug.

### Non-Converged Windows

There were `14` non-converged windows in total, covering `12` locations:

- `2023`: `8`
- `2024`: `5`
- `2025`: `1`

By type:

- `9` were `near_convergence_likely_postprocess_or_short_window`
- `2` were `historically_early_likely_training_coverage_gap`
- `3` were `requires_manual_review`

The only truly severe non-converged cases are still:

- `location_id` `103`, `2024-03-14`
- `location_id` `104`, `2024-03-14`

The other `9` near-converged windows are mostly not a major concern. Their
minimum `predicted_days` values were generally between `0.3` and `2.5`. They look
more like cases that almost reached zero within a short window, or cases where
post-processing was not aggressive enough.

These should be treated as explainable edge cases rather than top-priority
fixes.

The main uncertainty is limited to three manual-review windows:

- `location_id` `101`, `2023-04-24`
- `location_id` `103`, `2023-04-14`
- `location_id` `110`, `2023-04-14`

Current working interpretation:

- `101 / 2023` looks more like a local trajectory issue, with predictions rising
  again after the event date; its event-day GDD was also low, which may indicate
  weaker weather signals or insufficient feature representation
- `110 / 2023` was moderately late, with `day0_predicted_days` around `4.85`; it
  is not an extreme failure, but it is also not close to zero
- `103 / 2023` is the most important one to keep watching, because `104 / 2023`
  from the same date and area was already close to convergence while
  `103 / 2023` remained consistently `6-9` days late; this looks more like
  location-level generalization weakness than a pure year effect

The current working judgment is:

- the extreme failures at `103` and `104` in 2024 are a normal failure mode
  caused by limited training coverage; they matter for the business, but they do
  not look like an implementation bug
- most of the other non-converged cases are near-converged windows that did not
  fully reach zero
- the cases still worth deeper investigation are `101 / 2023`, `103 / 2023`, and
  `110 / 2023`

## Validation Performed

The following checks were run after the pipeline defaults were updated:

### Automated Tests

- `poetry run pytest tests/unit/services/test_post_event_zero_training.py tests/unit/pipeline/test_signed_target_outputs.py tests/regression/test_issue1_validation_helpers.py -q`
- `poetry run pytest tests/integration/test_backtesting_event_window_validation.py tests/unit/pipeline/test_signed_target_outputs.py -v`

These tests passed.

### CLI Smoke Checks

- `poetry run python -m app.pipeline.predict_only --help`
- `poetry run python -m app.pipeline.model_eval --help`

These commands completed successfully.

### Configuration Smoke Check

The pipeline objects were instantiated using the current default config to
confirm the runtime wiring:

- version: `v2.2`
- model type: `lightgbm`
- target: `signed_days_to_event`
- model manager: `LGBMModelManager`

This was verified for:

- `WeevilTrakPipeline`
- `BacktestingFramework`
- `model_eval` config loading

## Known Limitations

This handoff change updated defaults and validated the wiring, but it did not
re-run every full end-to-end production workflow.

In particular:

- a full `train_predict` run was not executed here
- a full `backtesting` run was not executed here
- a full `run_event_window_validation` run was not executed here

Those workflows depend on Redshift and weather data pulls and are much heavier
than a smoke check. The change was validated at the entry-point, config, and
signed-target behavior levels.

## Recommended Usage Going Forward

Unless there is a specific reason to compare with historical behavior, use:

- `app/config/weeviltrak_v2.2.yml`

Treat `v2.1` as:

- a legacy baseline
- a comparison target
- not the default production or analysis entry point

## Related Context

This update is part of the broader Stage 1 signed-target work. Related analysis
and result files live under:

- `docs/implemention_results/`

Those files describe the event-window validation, non-converged window review,
and Stage 1 anomaly analysis. This handoff note is intentionally narrower: it
documents the current pipeline defaults and what changed in code.

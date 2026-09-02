# WeevilTrak v2.4 Model Process Overview

## Scope

This is the current engineering architecture for `v2.4`. Older v2.1–v2.3
documents are historical comparison material, not default operating guidance.

## Outputs and targets

| ID | Business output | Training-label rule |
| ---: | --- | --- |
| 1 | Stage 1 | Observed Stage 1 |
| 4 | Phase 1 | Paired Stage 1/2 midpoint and eligible controlled Phase 1 labels |
| 2 | Stage 2 | Observed Stage 2 |
| 3 | Stage 3 / Phase 2 | Observed Stage 3 or Phase 2 |

The target is `signed_days_to_event`: positive before the event, zero on its
date, and negative after. `app/services/canonical_events.py` is mandatory;
raw Stage/Phase IDs are not stable across the 2026 transition.

## Data flow

```text
Redshift pest events + runtime manual events
  -> canonical Stage/Phase mapping
  -> gridded weather pull/cache
  -> causal seasonal feature engineering
  -> training matrix and post-event rows
  -> LightGBM Base Layer
  -> chronological OOF Base trajectories
  -> learned Ridge termination layer
  -> final Stage/Phase business output
```

Base features are `stage_id`, cumulative/rolling GDD and CDD, rolling humidity,
cumulative precipitation, DOY, latitude, and longitude.

## Two-layer behavior

LightGBM predicts signed days remaining and a Base event date. The Ridge
termination layer uses only causal trajectory history to calibrate and lock the
business-facing output. It is trained from chronological OOF Base trajectories,
not in-sample Base predictions. In the client release the first seven visible
days retain Base output; termination may publish from day eight.

## Training policy

The recent three-year window is intentional because earlier observations are
less reliable. v2.4 checks direct-label coverage for all four canonical IDs
before replacing artifacts. A failed gate is a safe stop: use the last approved
artifact in prediction-only mode rather than quietly weakening the policy.

## Code ownership

| Component | Location |
| --- | --- |
| Configuration | `app/config/weeviltrak_v2.4.yml` |
| Data/weather preparation | `app/services/data_preparation_service.py` |
| Canonical mapping | `app/services/canonical_events.py` |
| Manual events | `app/services/manual_events.py` |
| Managers/factory | `app/models/` |
| Train/predict | `app/pipeline/train_predict.py` |
| Termination | `app/pipeline/termination.py` |
| Backtesting | `app/pipeline/backtesting.py` |

See [model versions](../models/model_versions.md), the
[data contract](../data/historical_data_contract_and_risks.md), and the
[release runbook](../operations/v24_training_release_runbook.md).

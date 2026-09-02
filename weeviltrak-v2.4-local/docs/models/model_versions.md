# WeevilTrak Model Versions

This repository treats `v2.4` as the current default model configuration.

| Version | Config | Status | Model | Target | Notes |
| --- | --- | --- | --- | --- | --- |
| `v2.1` | `app/config/weeviltrak_v2.1.yml` | Legacy baseline | Random Forest | `days_to_event` | Kept for regression checks and RF vs LightGBM comparison. |
| `v2.2` | `app/config/weeviltrak_v2.2.yml` | Previous baseline | LightGBM | `signed_days_to_event` | Introduced signed targets, monotonic constraint, post-event zero rows, and zero-day termination. |
| `v2.3` | `app/config/weeviltrak_v2.3.yml` | Previous default | LightGBM | `signed_days_to_event` | Feature upgrade with `doy`, `cumu_cdd_air`, `rolling_cdd_air`, and a 14-day rolling window. |
| `v2.4` | `app/config/weeviltrak_v2.4.yml` | Current default | LightGBM + learned termination | `signed_days_to_event` | Canonical Stage/Phase pipeline. Directly models Stage 1, Phase 1, Stage 2, and Stage 3/Phase 2; Phase 1 training labels are midpoint(Stage 1, Stage 2). Uses a recent 3-year window and coverage gate. Ridge is trained only from chronological out-of-fold LightGBM trajectories and locks a trajectory after its first termination. |
| `v2.5` | `app/config/weeviltrak_v2.5.yml` | Deprecated compatibility config | LightGBM + learned termination | `signed_days_to_event` | Retained only so historical notebooks/config references do not break; do not create new production artifacts from it. |

Default scripted entry points should use `app/config/weeviltrak_v2.4.yml` unless they are intentionally comparing against older versions.

Local outputs should be written under `outputs/`:

- `outputs/backtesting/<version>/`
- `outputs/hindcast/<version>/`
- `outputs/business_checks/`
- `outputs/validation/`
- `outputs/cache/`
- `outputs/logs/`

# Data Contract

## Input rows

Each row represents one `prediction_date` × location × canonical `stage_id`.
Supply every one of IDs `1`, `4`, `2`, and `3` for each location/day.

| Field | Type | Description |
| --- | --- | --- |
| `prediction_date` | ISO date | Date on which the prediction is made |
| `location_id` | string | Stable location identifier |
| `place_id` | string, optional | Grid/place identifier for traceability |
| `latitude`, `longitude` | float | Site coordinates in decimal degrees |
| `stage_id` | integer | One of 1, 4, 2, 3 |
| `cumu_gdd_air` | float | Cumulative air growing degree days |
| `rolling_gdd_air` | float | Rolling air growing degree days |
| `cumu_cdd_air` | float | Cumulative air cooling degree days |
| `rolling_cdd_air` | float | Rolling air cooling degree days |
| `rolling_humidity_mean_pct` | float | Rolling mean relative humidity, percent |
| `cumu_precip_total_mm` | float | Cumulative precipitation, millimetres |
| `doy` | integer | Day of year, 1–366 |

The feature units and rolling-window calculations must match the release
configuration. Do not replace engineered inputs with raw daily temperature or
precipitation observations.

## Output rows

| Field | Meaning |
| --- | --- |
| `output_event_name` | Stage/Phase business name |
| `predicted_days` | Final business-facing days remaining |
| `predicted_stage_date` | Final business-facing predicted date |
| `base_predicted_days` | Base LightGBM days remaining before termination |
| `base_predicted_stage_date` | Base LightGBM event date before termination |
| `termination_history_days` | Number of visible prediction dates for this trajectory |
| `termination_output_applied` | Whether final output may use termination |
| `termination_reached` | Whether the termination decision has been reached |
| `termination_decision_date` | Date on which the event was locked, if reached |

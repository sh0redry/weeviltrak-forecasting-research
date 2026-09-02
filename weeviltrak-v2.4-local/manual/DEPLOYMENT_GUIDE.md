# Daily Deployment Guide

1. Update each active location's weather data.
2. Rebuild the configured cumulative and rolling features from the seasonal
   history through today.
3. Create four input rows per location/day, for stage IDs `1, 4, 2, 3`.
4. Append the new rows to the visible seasonal prediction history and run
   `predict_trajectory()`.
5. Publish only `predict_today()` results for the latest date.
6. Archive input and output CSVs with the release ID for traceability.

## Warm-up and termination

The termination model is evaluated from the first day, with missing early
history handled by its fitted imputers. To avoid treating sparse history as a
locked business decision, the first seven daily outputs retain Base LightGBM
predictions. Starting on the eighth visible day, the learned termination
output may replace the Base output. Once reached, the operational date remains
locked at zero remaining days for the rest of that trajectory.

## Input failures

Stop the run and correct the input when required features are missing, an
unsupported stage ID is supplied, dates cannot be parsed, or coordinates are
invalid. Do not silently fill missing engineered weather features with zero.

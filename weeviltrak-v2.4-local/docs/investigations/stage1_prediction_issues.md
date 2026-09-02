# WeevilTrak Model Prediction Issues - Diagnostic Report

**Date:** April 9, 2026
**Data source:** `weeviltrak_models_predictions_history` (DEV DB, PredictionDate 2026-03-01 to 2026-03-24)
**Scope:** Stage 1 (Overwintered Adults), US locations
**Diagnostic script:** `scripts/stage1_issues_diagnostic.py`
**Figures:** `data/plots/`

**Relevant files:** `app/services/data_preparation_service.py`, `app/models/model_manager.py`,
`app/config/weeviltrak_v2.1.yml`, `app/pipeline/train_predict.py`

---

## Issue 1 — PredictedDays Never Reaches Zero (No Stage-Reached State)

### Observation

Across all US locations below 45N, `PredictedDays` for Stage 1 **never reaches zero** during the Mar 1-24 window. The model counts down but stalls and reverses rather than completing.

- **KY (~37.9N):** min `PredictedDays` = 11 days (Mar 6)
- **WV (~38.6N):** min = 9 days (Mar 24)
- **Median minimum (below 45N):** 15 days

![Issue 1 — PredictedDays Never Reaches Zero](assets/issue1_never_reaches_zero.png)
*Figure 1: `PredictedDays` time series for all US locations (Mar 1–24, 2026). No location reaches zero.*

_Per-location prediction panels were not included in this repository distribution._
*Figure 1b: Per-location `PredictedDays` panels. All trajectories remain above zero throughout the window.*

### Current Resolution

Issue 1 is now addressed through a combination of modeling, training-data, and
validation changes. The fix is not a single line change. It is the result of
moving the default pipeline from the legacy `v2.1` Random Forest setup to the
current `v2.2` LightGBM signed-target workflow.

#### What `v2.2` Means

The current `v2.2` definition is:

- config: `app/config/weeviltrak_v2.2.yml`
- model type: `lightgbm`
- target: `signed_days_to_event`
- features:
  - `stage_id`
  - `cumu_gdd_air`
  - `rolling_gdd_air`
  - `rolling_humidity_mean_pct`
  - `cumu_precip_total_mm`
  - `latitude`
  - `longitude`
- monotonic constraint:
  - `cumu_gdd_air: -1`
- post-event training rows:
  - `post_event_training_days: 10`

In plain language, `v2.2` means the model is now trained with a signed event
distance target, uses LightGBM instead of Random Forest, and includes explicit
post-event samples during training.

#### Why `signed_days_to_event` Was Introduced

The older target, `days_to_event`, only describes the number of days remaining
until the event. It is non-negative by definition:

- before the event: positive
- on or after the event: zero

This makes the event boundary ambiguous during training. The model can learn
"smaller values mean closer to the event," but it does not naturally learn the
state transition from "not yet reached" to "already passed."

The new target, `signed_days_to_event`, makes that boundary explicit:

- before the event: positive
- on the event date: zero
- after the event: negative

This gives the model a full timeline around the event rather than only a
countdown that stops at zero.

The practical impact is:

- the model can better learn the event boundary
- analysis can inspect whether the raw model output has crossed the event
- business-facing output still remains compatible with the old countdown format

#### How the Output Works Now

The pipeline now keeps two related outputs:

- `raw_signed_days`: direct model output from the signed target
- `predicted_days`: business-facing countdown, defined as
  `max(raw_signed_days, 0)`

So the model is allowed to learn that an event has already happened, while the
external countdown field still behaves like "days remaining" and never goes
below zero.

#### Training Data Change

The training pipeline now includes post-event rows instead of training only on
future-event examples.

That means the model no longer sees only:

- 12 days before the event
- 8 days before the event
- 3 days before the event

It also sees:

- 0 days at the event boundary
- negative signed values after the event

This is the key reason the model can now learn a completed-state boundary rather
than only a positive countdown.

#### Model Change

The default model is now LightGBM rather than Random Forest.

This matters for Issue 1 because:

- Random Forest had no structural mechanism to encourage event-boundary
  consistency
- LightGBM in `v2.2` uses a monotonic constraint on `cumu_gdd_air`
- this reduces the chance that predictions move away from the event as thermal
  accumulation increases

This does not make the model perfect, but it removes one major source of the
old "count down, stall, then drift away again" behavior.

### Validation Result

After the signed-target update, Issue 1 was re-evaluated using an
event-centered validation window:

- `prediction_date in [stage_date - 3, stage_date + 3]`

This validation is more appropriate than the earlier long window when the goal
is to answer a specific question:

- does the model terminate near the actual event date?

Under this updated validation:

- total event windows: `80`
- windows with `predicted_days <= 0` within `[-3, +3]`: `66`
- window-level termination rate: `82.5%`

This changes the interpretation of Issue 1 in an important way.

The current evidence does **not** support the old claim that Stage 1 predictions
generally "never reach zero." Instead, the signed-target workflow shows that the
majority of windows do terminate near the event date.

### Remaining Limitations

Issue 1 is no longer a general system-wide failure, but a smaller set of
residual cases still exists.

The remaining non-coverage cases fall into three broad groups:

1. **Near-convergence windows**
   - predictions are already very close to zero
   - these are likely short-window or post-processing edge cases

2. **Historically early events**
   - the real event occurs much earlier than the location's training history
   - this is mainly a training-coverage limitation rather than a logic failure

3. **A few location-level generalization problems**
   - these are isolated cases where the model remains too late for a specific
     site even without an obvious extreme-early explanation

So the current answer to Issue 1 is:

- the original failure mode was mitigated by the `v2.2` signed-target LightGBM
  workflow
- the main problem is no longer "PredictedDays never reaches zero"
- the remaining work is focused on edge cases, training coverage, and local
  generalization quality rather than a universal termination failure

---

## Issue 2 — Cold Snap Sensitivity Causes ±15-25 Day Swings

### Observation

A cold snap around **Mar 7-14** caused `PredictedDays` to swing ±15-25 days within one week:

1. **Mar 4-7 (warm):** GDD accumulated rapidly — predictions dropped 12-15 days/day for southern bands
2. **Mar 7-14 (cold snap):** GDD stalled — predictions reversed +4-5 days/day, pushing dates back 15-25 days

**Net drift by Mar 24 vs Mar 1:**

| Lat Band | Net Drift | Direction |
|---|:---:|---|
| 33-35N | +1.6d | Later |
| 35-37N | +3.4d | Later |
| 37-39N | +8.4d | **Later** |
| 39-40N | -7.8d | Earlier |
| 40-42N | -9.7d | Earlier |

![Issue 2 — Cold Snap Sensitivity](assets/issue2_forecast_drift_reversal.png)
*Figure 2: Daily `PredictedDays` by latitude band. The warm-then-cold sequence produces ±15-25 day swings.*

_The predicted-stage-date drift figure was not included in this repository distribution._
*Figure 2b: `PredictedStageDate` drift over the observation window.*

### Root Cause

The 7-day `rolling_gdd_air` window is short enough that a single cold week dominates the feature. Southern locations near their GDD event threshold are in a steep region of the model's response surface — small feature changes produce large output swings. RandomForest has no gradient-based regularization, so a cold week can push features into a sparse interpolation region where the forest averages unrelated leaf values. No `doy` feature exists to anchor seasonal position.

LightGBM addresses this through regularization (`learning_rate`, `min_child_samples`, `max_depth`) and monotonic constraints — a cold snap that *slows* GDD accumulation (without reversing it) cannot cause predictions to increase.

### Training Fix

**A — Extend rolling GDD window from 7 to 14 days**

```python
# data_preparation_service.py :: _calc_rolling_avg()
ROLLING_WINDOW_DAYS = 14   # was: 7

weather_df["rolling_gdd_air"] = (
    weather_df
    .groupby(["pest_year", "place_id"])["gdd_air"]
    .transform(lambda x: x.rolling(ROLLING_WINDOW_DAYS, min_periods=1).mean())
)
```

Add `rolling_window_days: 14` to config. Regenerate all training data with the new window before fitting — do not mix 7-day training features with 14-day inference features.

**B — Add `doy` (day-of-year) feature**

Strongest stable seasonal anchor. Without it, the model cannot tell whether cumulative GDD is ahead or behind for the time of year:

```python
# data_preparation_service.py :: process_weather_data()
weather_df["doy"] = weather_df["date"].dt.dayofyear
```

Update config features to include `doy`.

**C — Add cold-degree-days (CDD) features**

GDD clips cold days to zero, so the model has no signal about cold intensity — it cannot distinguish a mild 9°F plateau from a hard freeze at -5°F. CDD accumulates cold below the same 50°F base, giving the model direct signal about cold stress:

$$\text{CDD}_{daily} = \max\!\big(0,\ 50\text{°F} - \frac{T_{max} + T_{min}}{2}\big)$$

```python
# data_preparation_service.py :: process_weather_data()
CDD_BASE_TEMP = 50  # same base as GDD, in Fahrenheit

weather_df["cdd_air"] = (CDD_BASE_TEMP - (df["air_temp_min_f"] + df["air_temp_max_f"]) / 2).clip(lower=0)

weather_df["cumu_cdd_air"] = (
    weather_df
    .groupby(["pest_year", "place_id"])["cdd_air"]
    .cumsum()
)

weather_df["rolling_cdd_air"] = (
    weather_df
    .groupby(["pest_year", "place_id"])["cdd_air"]
    .transform(lambda x: x.rolling(14, min_periods=1).mean())
)
```

A location with 50 cumulative GDD and 0 CDD (mild start, slow warming) is very different from one with 50 GDD and 30 CDD (warm days interrupted by freezes). `rolling_cdd_air` directly spikes during events like the Mar 7-14 cold snap, letting the model learn that "high recent CDD = temporary delay, don't overreact" vs "low CDD + low GDD = genuinely slow season."

---

## Issue 3 — Systematic Late Bias for Southern Locations

### Observation

On Mar 24, the model predicts Stage 1 **8-13 days later** than 2018-2025 historical median for southern lats, and **11-13 days early** for northern lats — an inverted latitudinal gradient:

| Lat Band | Model (Mar 24) | Historical | Bias |
|---|:---:|:---:|:---:|
| 33-35N | Apr 13 | Apr 5 | **+8d late** |
| 35-37N | Apr 21 | Apr 8 | **+13d late** |
| 37-39N | Apr 20 | Apr 11 | **+9d late** |
| 39-40N | Apr 7 | Apr 20 | -13d early |
| 40-42N | Apr 13 | Apr 23 | -11d early |

![Issue 3 — Late Bias](assets/issue3_late_bias_vs_historical.png)
*Figure 3: Predicted vs historical stage dates by latitude band. Southern bands 8-13d late; northern bands 11-13d early.*

_The latitude-gradient inversion figure was not included in this repository distribution._
*Figure 4: Latitudinal gradient on Mar 1 (blue) vs Mar 24 (orange).*

### Root Cause

1. **Lat-band imbalance:** Southern locations (KY, OH, VA, NC) have 5 seasons (2021+) vs 8 for northeastern sites. ~62% of training rows are northern.
2. **No inter-annual GDD context:** Raw `cumu_gdd_air` doesn't tell the model whether this is a warm or cold year relative to climatology.
3. **Cold snap residual from Issue 2:** Disproportionately suppressed GDD for southern lats near their event threshold.

### Training Fix

**A — Add `gdd_anomaly` ratio feature**

Ratio of current cumulative GDD to per-location, per-DOY climatological mean — tells the model whether the season is running warm or cold:

```python
# data_preparation_service.py :: process_weather_data()
clim_gdd = (
    weather_df[weather_df["pest_year"] < current_year]
    .groupby(["place_id", "doy"])["cumu_gdd_air"]
    .mean()
    .rename("clim_cumu_gdd_air")
    .reset_index()
)
weather_df = weather_df.merge(clim_gdd, on=["place_id", "doy"], how="left")
weather_df["gdd_anomaly"] = (
    weather_df["cumu_gdd_air"] /
    weather_df["clim_cumu_gdd_air"].replace(0, np.nan)
).fillna(1.0).clip(0.3, 3.0)
```

**B — Stratified lat-band resampling**

Oversample under-represented lat bands to match the most populous band:

```python
# data_preparation_service.py :: prepare_training_data()
from sklearn.utils import resample

def balance_lat_bands(df, lat_col="latitude"):
    bands = [(33, 38.5), (38.5, 42), (42, 46)]
    groups = [df[(df[lat_col] >= lo) & (df[lat_col] < hi)] for lo, hi in bands]
    target_n = max(len(g) for g in groups)
    balanced = [
        resample(g, replace=True, n_samples=target_n, random_state=42)
        if len(g) < target_n else g
        for g in groups
    ]
    return pd.concat(balanced, ignore_index=True).sample(frac=1, random_state=42)

model_data = balance_lat_bands(model_data)
```

---

## Retraining Plan

### Updated Feature Set

### Feature Interpretations

| Feature | What it captures | Computation | Units |
|---------|-----------------|-------------|-------|
| `stage_id` | Which life stage is being predicted (1=Overwintered Adults, 2=F1 Eggs, 3=F1 Adults) | Categorical from trap data | Integer |
| `cumu_gdd_air` | Total heat accumulated since season start — primary driver of phenological progress | $\sum \max(0,\ \frac{T_{max}+T_{min}}{2} - 50\text{°F})$ from Mar 1 | Degree-days (°F) |
| `rolling_gdd_air` | Recent heat trend — captures whether warmth is accelerating or stalling | 14-day rolling mean of daily GDD | Degree-days/day |
| `rolling_humidity_mean_pct` | Recent moisture conditions — humidity affects insect activity and development rate | 7-day rolling mean of daily mean humidity | Percent |
| `cumu_precip_total_mm` | Total precipitation since season start — soil moisture affects overwintering survival | Cumulative daily precip from Mar 1 | mm |
| `latitude` | North-south position — controls base phenological timing (south = earlier) | From grid centroid | Decimal degrees |
| `longitude` | East-west position — captures coastal vs inland climate effects | From grid centroid | Decimal degrees |
| `doy` | **NEW.** Day of year — stable seasonal anchor independent of weather | `date.dayofyear` | 1-366 |
| `gdd_anomaly` | **NEW.** Whether this season is running warm or cold vs climatology | `cumu_gdd_air / mean(cumu_gdd_air at same place+DOY, prior years)` | Ratio (1.0 = normal) |
| `cumu_cdd_air` | **NEW.** Total cold stress since season start — distinguishes mild from freezing conditions | $\sum \max(0,\ 50\text{°F} - \frac{T_{max}+T_{min}}{2})$ from Mar 1 | Degree-days (°F) |
| `rolling_cdd_air` | **NEW.** Recent cold intensity — directly encodes cold snap events | 14-day rolling mean of daily CDD | Degree-days/day |

### Why LightGBM Over RandomForest

| Concern | RandomForest (current) | LightGBM |
|---------|----------------------|----------|
| Monotonic constraints | Not supported — predictions can bounce back | Native `monotone_constraints` per feature — fixes Issues 1+2 at model level |
| Regularization | No gradient signal — averages unrelated leaf values in sparse regions | `learning_rate`, `min_child_samples`, `max_depth` provide inherent smoothing |
| Feature utilization | Random subsets per split (`√n ≈ 3` of 11 features) — new features may be underused | Gradient-based splits efficiently find the best feature at each node |
| Tuning | Simpler — fewer knobs | Needs `learning_rate`, `max_depth`, `num_leaves` but params below are a solid starting point |
| Overfitting risk | Low inherently | Comparable with `subsample=0.8`, `min_child_samples=20` |
| Validation | OOB error | Requires explicit validation — use existing `BacktestingFramework` |

```yaml
# weeviltrak_v2.1.yml
model_type: lightgbm
features:
  - stage_id
  - cumu_gdd_air
  - rolling_gdd_air          # 14-day window (was 7)
  - rolling_humidity_mean_pct
  - cumu_precip_total_mm
  - latitude
  - longitude
  - doy                      # NEW
  - gdd_anomaly              # NEW
  - cumu_cdd_air             # NEW — cumulative cold stress
  - rolling_cdd_air          # NEW — recent cold intensity (14-day)
target: days_to_event         # now includes 0 (post-event rows)
post_event_window_days: 30
rolling_window_days: 14
model_parameters:
  learning_rate: 0.05
  max_depth: 6
  num_leaves: 31
  min_child_samples: 20
  subsample: 0.8
  colsample_bytree: 0.8
  n_estimators: 200
  random_state: 42
  n_jobs: 5
  monotone_constraints: [0, -1, 0, 0, -1, 0, 0, 0, 0, 1, 0]
  # [stage_id, cumu_gdd_air, rolling_gdd_air, rolling_humidity_mean_pct,
  #  cumu_precip_total_mm, latitude, longitude, doy, gdd_anomaly,
  #  cumu_cdd_air, rolling_cdd_air]
  # -1 = more GDD/precip → fewer days remaining
  # +1 = more cold stress → more days remaining
```

### Sequencing

All changes applied in a single retraining run. Regenerate features first, then fit.

| Step | Change | Files | Addresses |
|------|--------|-------|-----------|
| **1** | Switch model from RandomForest to LightGBM with monotonic constraints | `model_manager.py`, config | Issues 1+2 |
| **2** | Add post-event rows (`days_to_event=0`) via backward merge | `data_preparation_service.py`, config | Issue 1 |
| **3** | Extend rolling GDD window 7→14 days | `data_preparation_service.py`, config | Issue 2 |
| **4** | Add `doy` feature | `data_preparation_service.py`, config | Issues 2+3 |
| **5** | Add CDD features (`cumu_cdd_air`, `rolling_cdd_air`) | `data_preparation_service.py`, config | Issue 2 |
| **6** | Add `gdd_anomaly` ratio feature | `data_preparation_service.py`, config | Issue 3 |
| **7** | Lat-band stratified resampling | `data_preparation_service.py` | Issue 3 |

After retraining, validate with backtesting (`BacktestingFramework`) before deploying to production.

Dependency: `lightgbm==4.6.0` (already installed).

---

## Future: Spatial GP Add-On (v2.2)

If backtesting after the LightGBM retraining still shows systematic location-specific residuals (especially the southern late bias), a Spatial GP layer can be added on top. Full spec: [`docs/models/lgbm_gp_model_plan.md`](../models/lgbm_gp_model_plan.md).

The GP fits on per-location training residuals from LightGBM. Southern locations with systematic bias get a correction at inference. GP sigma provides calibrated confidence intervals — wider for under-represented locations, narrower for well-observed sites. New locations revert to pure LightGBM.

```
LightGBM predictions
    → Compute training residuals per (lat, lon)
    → Fit Spatial GP (Matern kernel, 2D km coordinates)
    → y_final = y_lgbm + y_gp
    → sigma → confidence intervals
```

This is **not needed for the initial retraining** — the feature engineering fixes (`gdd_anomaly`, CDD, lat-band resampling) plus LightGBM's regularization may already resolve the southern bias. Evaluate with backtesting first, then add the GP only if residual spatial structure remains.

| Step | File | Change |
|------|------|--------|
| 1 | `app/models/lgbm_spatial_gp.py` *(new)* | `LGBMSpatialGPModel`: `.fit()`, `.predict()`, `.predict_ci()` |
| 2 | `app/models/model_manager.py` | `LGBMSpatialGPModelManager` wrapping the estimator |
| 3 | `app/config/weeviltrak_v2.2.yml` *(new)* | `model_type: lgbm_gp` config |
| 4 | `app/pipeline/train_predict.py` | Use `create_model_manager(config, s3_manager)` factory |

Dependency: `sklearn.gaussian_process` (bundled with scikit-learn).

---

*Diagnostic conducted April 9, 2026 from DEV DB history table (Mar 1-24, 2026).*
*Analysis script: `scripts/stage1_issues_diagnostic.py`. Figures: `data/plots/`.*

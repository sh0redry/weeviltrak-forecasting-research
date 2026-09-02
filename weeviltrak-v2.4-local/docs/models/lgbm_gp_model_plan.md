# Two-Stage Model: LightGBM + Spatial GP Residual Kriging

## Motivation

The current Random Forest model uses `latitude` and `longitude` as features but cannot extrapolate beyond the spatial bounds of training data — it returns the nearest training cluster's average for unseen locations. A two-stage approach separates the **weather/phenology response** (handled by LightGBM) from **spatial bias correction** (handled by a Gaussian Process on residuals), giving proper spatial extrapolation and calibrated uncertainty.

---

## Model Architecture

### Stage 1 — LightGBM (primary learner)

- Trained on all 7 features: `stage_id`, `cumu_gdd_air`, `rolling_gdd_air`, `rolling_humidity_mean_pct`, `cumu_precip_total_mm`, `latitude`, `longitude`
- Predicts `days_to_event` directly
- Replaces RF: better regularization, faster training, native missing-value handling

### Stage 2 — Spatial GP on residuals (spatial correction)

- Compute training residuals: `r_i = y_i - ŷ_i_lgbm`
- **Aggregate residuals per unique (lat, lon)** — one mean residual per location
  - This keeps the GP input small (~tens to low hundreds of unique locations), so vanilla `sklearn.gaussian_process.GaussianProcessRegressor` is sufficient — no sparse approximation needed
- Fit GP with Matérn(ν=5/2) + WhiteKernel on 2D (lat, lon) → mean residual
- At prediction time: `ŷ = ŷ_lgbm + ŷ_gp`
- GP also provides `σ` per point → natural confidence intervals (replaces OOB CI hack)

---

## Implementation Steps

### Step 1: Refactor `model_manager.py` into a pluggable architecture

**File**: `app/models/model_manager.py`

1. Define a base protocol/ABC with the interface:
   - `train(x_train, y_train, **kwargs) → model`
   - `predict(model, x_test, y_train, alpha) → pd.DataFrame`
   - `save_model(model, model_path)`
   - `load_model(model_path) → model`

2. Rename the existing class to `RFModelManager` — preserves all current RF logic unchanged.

3. Add a factory function:
   ```python
   def create_model_manager(config, s3_manager) -> ModelManager:
       model_type = config.config.get("model_type", "rf")
       if model_type == "rf":
           return RFModelManager(config, s3_manager)
       elif model_type == "lgbm_gp":
           return LGBMSpatialGPModelManager(config, s3_manager)
       else:
           raise ValueError(f"Unknown model_type: {model_type}")
   ```

### Step 2: Implement `LGBMSpatialGPModel` (sklearn-compatible estimator)

**File**: `app/models/lgbm_spatial_gp.py` (new file)

A single class that wraps both stages and exposes `.fit()` / `.predict()` — drop-in compatible with how the pipeline uses the RF model today.

```python
class LGBMSpatialGPModel:
    """
    Two-stage estimator: LightGBM + Spatial GP residual kriging.
    
    Follows the same interface as sklearn estimators:
      - .fit(X, y)      → trains both stages, returns self
      - .predict(X)      → returns combined predictions (LightGBM + GP correction)
      - .predict_ci(X, alpha) → returns (pred, lower, upper)
    
    Pickles as a single object — same save/load flow as RandomForestRegressor.
    """
    
    def __init__(self, lgbm_params=None, spatial_cols=None, **kwargs):
        ...
    
    def fit(self, X, y):
        # 1. Fit LightGBM on all features (with monotonic constraints from lgbm_params)
        # 2. Compute in-sample residuals
        # 3. Aggregate residuals by unique (lat, lon)
        # 4. Project (lat, lon) to km; store coord_origin_
        # 5. Fit GP on projected coords → mean residual
        # 6. Store lgbm_, gp_, coord_origin_ as attributes
        return self
    
    def predict(self, X):
        # 1. y_lgbm = self.lgbm_.predict(X)
        # 2. Deduplicate (lat, lon), project to km, GP predict (mean only)
        # 3. Broadcast GP corrections back to all rows
        # 4. return y_lgbm + gp_correction
        ...
    
    def predict_ci(self, X, alpha=0.05):
        # Same as predict() but also returns lower/upper bounds
        # using GP std and z = norm.ppf(1 - alpha/2)
        ...
```

This means:
- **`model_manager.train()`** calls `model.fit(x_train, y_train)` and returns the model — same pattern as RF
- **`model_manager.save_model(model)`** pickles the single `LGBMSpatialGPModel` object — same as RF
- **`model_manager.load_model()`** unpickles it — same as RF
- **`model.predict(x_test[features])`** works identically to `RandomForestRegressor.predict()` — the pipeline's direct `model.predict()` call on line ~216 of `train_predict.py` works without changes

### Step 2b: Implement `LGBMSpatialGPModelManager`

**File**: `app/models/model_manager.py` (same file, new class)

The model manager wraps the estimator but keeps the same `train/predict/save_model/load_model` interface as `RFModelManager`:

#### `train(x_train, y_train, **kwargs)` → `LGBMSpatialGPModel`

1. Extract LightGBM params from `self.config.model_params` (with sensible defaults for LightGBM).
2. Apply **monotonic constraints** from config (see [Monotonic Constraints](#monotonic-constraints-on-cumulative-features) below). These enforce that `days_to_event` can only decrease as cumulative GDD and precipitation increase.
3. Create `LGBMSpatialGPModel(lgbm_params=model_params, spatial_cols=["latitude", "longitude"])`.
4. Call `model.fit(x_train, y_train.values.ravel())`.
5. Store and return `self.model` — a single picklable object, same as RF.

#### `predict(model, x_test, y_train, alpha)` → `pd.DataFrame`

1. `y_pred = model.predict(x_test[features])`  ← identical call to RF path
2. If `alpha` is provided: `y_pred, y_lower, y_upper = model.predict_ci(x_test[features], alpha)`
3. **Apply zero-day termination** (see [Post-Processing: Zero-Day Termination](#post-processing-zero-day-termination) below).
4. Assemble output DataFrame (same format as current RF output).

#### `save_model / load_model`

- Pickle the single `LGBMSpatialGPModel` object — identical flow to `RandomForestRegressor`. No special handling needed.

### Step 3: Add config support

**File**: `app/config/weeviltrak_v2.1.yml`

- Add `model_type: "rf"` to preserve backward compatibility (existing config continues to use RF).

**New file**: `app/config/weeviltrak_v2.2.yml`

- Copy of v2.1 with:
  ```yaml
  model_type: "lgbm_gp"
  model_parameters:
    n_estimators: 200
    learning_rate: 0.05
    max_depth: 6
    num_leaves: 31
    min_child_samples: 20
    subsample: 0.8
    colsample_bytree: 0.8
    random_state: 42
    n_jobs: 5
    # Monotonic constraints per feature (order matches features list):
    #   stage_id=0, cumu_gdd_air=-1, rolling_gdd_air=0,
    #   rolling_humidity_mean_pct=0, cumu_precip_total_mm=-1,
    #   latitude=0, longitude=0
    # -1 = decreasing: more accumulated GDD/precip → fewer days remaining
    monotone_constraints: [0, -1, 0, 0, -1, 0, 0]
  model_path: "model_testing_2025/model_weeviltrak_lgbm_gp.pkl"
  ```

### Step 4: Update pipeline to use factory

**File**: `app/pipeline/train_predict.py`

1. Change import:
   ```python
   from app.models.model_manager import create_model_manager
   ```
2. In `__init__`, replace:
   ```python
   self.model_manager = ModelManager(self.config, self.s3_manager)
   ```
   with:
   ```python
   self.model_manager = create_model_manager(self.config, self.s3_manager)
   ```
3. Fix `run_prediction` line ~216 where `model.predict(x_test[features])` is called directly on the loaded model object — route through `self.model_manager.predict()` instead, so the GP correction is applied.

**File**: `app/pipeline/predict_only.py`

- No change needed (already uses `pipeline.model_manager.load_model()`).

### Step 5: Verify

1. Run import checks:
   ```bash
   poetry run python -c "from app.models.model_manager import create_model_manager; print('OK')"
   ```
2. Run training with v2.1 config (RF) — should behave identically to before.
3. Run training with v2.2 config (LGBM+GP) — new model.
4. Compare predictions on backtesting set.

---

## What Stays the Same

| Component | Status |
|-----------|--------|
| Feature engineering (`DataPreparationService`) | Unchanged |
| Config `features` list | Unchanged |
| Weather data pipeline | Unchanged |
| S3 model storage format | Unchanged (pickle) |
| Output DataFrame format | Unchanged |
| `predict_only.py` | Unchanged |

## New Dependencies

| Package | Version | Notes |
|---------|---------|-------|
| `lightgbm` | 4.6.0 | Already installed |
| `sklearn.gaussian_process` | (bundled with scikit-learn) | Already available |
| `scipy.stats` | (bundled with scipy) | Already available |

## Risk / Notes

- The GP aggregates residuals **per unique (lat, lon)**, not per row. This keeps the covariance matrix small and tractable. With ~50–200 unique locations in the weevil dataset, sklearn's exact GP is fast enough.
- The GP's Matérn kernel smoothly interpolates and extrapolates spatial residuals — new locations get a correction based on distance to known locations, decaying gracefully to zero (i.e., pure LightGBM prediction) far from any training site.
- Confidence intervals from the GP are calibrated — wider for locations far from training data, narrower for well-observed locations. This is a significant improvement over the OOB-based constant-width CI.

---

## Monotonic Constraints on Cumulative Features

### Problem

The model predicts `days_to_event` — the number of days until a pest stage is reached. Physically, as the season progresses and heat/moisture accumulate, the number of days remaining can only decrease (the event gets closer, not further away). However, tree-based models have no built-in notion of this — a random split on `cumu_gdd_air` could assign a *higher* prediction to a *higher* GDD bucket. This causes "bounce-back" artifacts where predictions increase after the stage date has passed.

### Solution: LightGBM `monotone_constraints`

LightGBM natively supports monotonic constraints per feature. During tree construction, splits are restricted so the predicted value can only move in the specified direction as the feature increases.

For our feature set `[stage_id, cumu_gdd_air, rolling_gdd_air, rolling_humidity_mean_pct, cumu_precip_total_mm, latitude, longitude]`:

```yaml
monotone_constraints: [0, -1, 0, 0, -1, 0, 0]
```

| Feature | Constraint | Meaning |
|---------|-----------|---------|
| `stage_id` | 0 (none) | Different stages have different magnitude — no monotonic relationship |
| `cumu_gdd_air` | **-1 (decreasing)** | More accumulated GDD → fewer days remaining |
| `rolling_gdd_air` | 0 (none) | Short-term rolling average can go up or down |
| `rolling_humidity_mean_pct` | 0 (none) | No a priori monotonic relationship |
| `cumu_precip_total_mm` | **-1 (decreasing)** | More accumulated precipitation → fewer days remaining |
| `latitude` | 0 (none) | No strict monotonic spatial relationship |
| `longitude` | 0 (none) | No strict monotonic spatial relationship |

### Why this works

- `cumu_gdd_air` and `cumu_precip_total_mm` are **cumulative counters** — they only increase through the season by construction
- By constraining the model to be monotonically decreasing in these features, **predictions can only decrease (or stay flat) as time progresses**
- This directly encodes the physical law: "more accumulated heat → closer to / past the event"
- Once a location accumulates enough GDD, the prediction is forced toward zero and cannot bounce back up

### Relationship to zero-day termination

Monotonic constraints handle ~95% of the post-event bounce-back problem at the model level. The two-consecutive-day termination rule (see [Post-Processing: Zero-Day Termination](#post-processing-zero-day-termination)) is kept as a **safety net** for edge cases where:
- GDD barely changes day-to-day (stalled weather)
- The model plateaus just above zero without crossing the threshold

The two mechanisms are complementary: constraints prevent the model from generating bad predictions in the first place; termination catches any remaining edge cases in post-processing.

---

## Spatial Coordinate Handling

### Problem

Raw latitude/longitude in degrees distort distances at high latitudes. The prediction domain spans ~38°N (southern Pennsylvania) to ~52°N (northern Ontario/Quebec), and ~65°W to ~85°W. One degree of longitude at 38°N ≈ 88 km, but at 52°N ≈ 69 km — a ~22% distortion. The GP kernel uses Euclidean distance, so fitting on raw degrees would systematically misweight east-west vs. north-south distances.

### Solution: Equirectangular projection to km

Convert (lat, lon) to approximate (x_km, y_km) relative to a reference point (centroid of training locations):

```python
import numpy as np

def latlon_to_km(lat, lon, ref_lat, ref_lon):
    """Equirectangular projection — accurate enough for <2000 km spans."""
    R = 6371.0  # Earth radius in km
    x_km = R * np.radians(lon - ref_lon) * np.cos(np.radians(ref_lat))
    y_km = R * np.radians(lat - ref_lat)
    return x_km, y_km
```

- `ref_lat, ref_lon` = centroid of training locations (stored in model dict so prediction uses the same origin)
- This is simpler and faster than a full UTM projection, and accurate to <1% over the domain
- Both `fit()` and `predict()` use the same projection

### Why not haversine kernel directly?

scikit-learn's `GaussianProcessRegressor` only supports stationary kernels with Euclidean distance. A custom haversine kernel would work but complicates serialization and is unnecessary when the equirectangular projection is accurate enough for this domain.

---

## Production-Scale Prediction

### Problem

In production, predictions cover the entire northeast US + Ontario + Quebec at 10×10 km resolution. That's **~5,000–10,000 unique grid cells**, each with multiple date-rows (one per day of the season). Total prediction rows can reach **hundreds of thousands to millions**.

### Solution: Deduplicated GP prediction with broadcast

The GP correction depends only on (lat, lon), not on date or weather features. So:

1. **Extract unique (lat, lon)** from `x_test`:
   ```python
   unique_coords = x_test[["latitude", "longitude"]].drop_duplicates()
   ```
2. **Project to km** using the stored `coord_origin`.
3. **Call `gp.predict()` once** on the ~5,000–10,000 unique coordinates:
   ```python
   gp_mean, gp_std = model["gp"].predict(unique_coords_km, return_std=True)
   ```
4. **Join corrections back** to the full `x_test` by (lat, lon):
   ```python
   unique_coords["gp_mean"] = gp_mean
   unique_coords["gp_std"] = gp_std
   x_test = x_test.merge(unique_coords, on=["latitude", "longitude"], how="left")
   ```

### Scaling analysis

| Component | Training (n_train_locs ≈ 200) | Prediction (n_pred_locs ≈ 10,000) |
|-----------|-------------------------------|-------------------------------------|
| GP `fit()` | O(200³) ≈ 8M ops → instant | — |
| GP `predict()` | — | O(200² × 10,000) ≈ 400M ops → ~1–5 seconds |
| LightGBM `predict()` | — | O(n_rows × n_trees) → seconds |
| Total extra cost vs. RF | Negligible | ~5 seconds overhead for GP |

### Behavior far from training sites

For grid cells far from any training observation (e.g., northern Ontario):
- GP mean correction → **0** (reverts to prior mean)
- GP std → **large** (high uncertainty)
- Net prediction → **pure LightGBM** with wide confidence interval

This is the correct, honest behavior: the model says "I have no spatial bias data for this area, so I rely on weather features alone and flag high uncertainty."

---

## Post-Processing: Zero-Day Termination

### Problem

The model predicts `days_to_event` — the number of days until a given pest stage is reached. Once the stage date has actually passed, the true value is 0 (the event already happened). However, the current model sometimes predicts values > 0 even after the stage date, because neither RF nor LightGBM has an explicit "already happened" mechanism.

### Solution: Two-consecutive-day termination rule

For each **(location, stage_id)** time series sorted by date:

1. Scan forward through the daily predictions.
2. If **two consecutive days** both have `pred < 1.0`, declare the event as reached.
3. From that first triggering day onward, set `pred = 0` (and `pred_lower = 0`, `pred_upper = 0` if CIs exist).

```python
def apply_zero_day_termination(pred_df: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    """
    For each (location, stage_id) time series: once two consecutive days
    predict < threshold, set predictions to 0 from the first triggering day onward.

    Args:
        pred_df: DataFrame with columns including 'pred', 'stage_id',
                 'latitude', 'longitude', and a DatetimeIndex (date).
        threshold: Consecutive-day threshold (default 1.0).

    Returns:
        DataFrame with terminated predictions zeroed out.
    """
    df = pred_df.copy()
    group_cols = ["latitude", "longitude", "stage_id"]

    for _, group in df.groupby(group_cols):
        idx = group.sort_index().index  # sorted by date
        preds = df.loc[idx, "pred"].values
        below = preds < threshold
        for i in range(1, len(below)):
            if below[i - 1] and below[i]:
                # Two consecutive days below threshold — zero out from day i-1 onward
                zero_idx = idx[i - 1:]
                df.loc[zero_idx, "pred"] = 0.0
                if "pred_lower" in df.columns:
                    df.loc[zero_idx, "pred_lower"] = 0.0
                if "pred_upper" in df.columns:
                    df.loc[zero_idx, "pred_upper"] = 0.0
                break
    return df
```

### Where this lives

- Implemented as a static/utility method on the model manager (or a standalone function in `model_manager.py`).
- Called inside `predict()` **after** combining LightGBM + GP corrections and **before** assembling the final output DataFrame.
- Applied identically regardless of model type (RF or LGBM+GP) — this is a domain rule, not model-specific.

### Why two consecutive days?

A single day below 1.0 could be noise (e.g., a brief temperature drop causing the model to briefly dip). Requiring two consecutive days filters transient dips while still detecting the genuine transition to "event reached." The threshold of 1.0 is conservative: a prediction of < 1 day essentially means "the event is happening today or already happened."

### Edge case: prediction never triggers

If a location/stage never has two consecutive days < 1.0 within the prediction window, no zeroing is applied — the raw model predictions stand as-is. This is the correct behavior for stages that haven't been reached yet.

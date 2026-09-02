# LightGBM Monotonic Constraints V1

## Purpose

This document defines the first implementation slice of the LightGBM migration: add monotonic-constraint support to the LightGBM primary model only.

This is intentionally narrower than the full plan in [lgbm_gp_model_plan.md](lgbm_gp_model_plan.md). It does not include GP residual correction, confidence intervals, zero-day termination, or backtesting work.

## Scope for V1

V1 includes only the following:

- Add config support for monotonic constraints inside `model_parameters`
- Express constraints by feature name, not only by positional list
- Support only one active constraint in the first version:
  - `cumu_gdd_air: -1`
- Convert feature-name constraints into the ordered `monotone_constraints` list required by LightGBM
- Apply the generated constraint list only on the LightGBM model path
- Add a minimal local monotonicity verification

V1 explicitly does not include:

- GP residual kriging
- Prediction intervals
- Zero-day termination
- Backtesting-based validation
- Constraints on `cumu_precip_total_mm`

## Why Start Here

This is a low-risk and high-signal first step.

- The change is localized to the LightGBM training path
- The business meaning is clear: as cumulative GDD increases, `days_to_event` should not increase
- It lets us validate the monotonic-constraint mechanism before adding GP complexity
- It avoids mixing model-structure validation with larger experimental work

## Intended Semantics

The constraint is defined at the LightGBM model level, not at the pipeline level.

For V1, the requirement is:

- Holding all other features fixed, increasing `cumu_gdd_air` must not increase the predicted `days_to_event`

This does **not** mean:

- Real day-by-day forecasts for a location must always strictly decrease

That stronger statement is not guaranteed, because other input features also change over time, including rolling features and cumulative precipitation.

## Config Design

The monotonic-constraint declaration lives inside `model_parameters`.

Recommended config shape:

```yaml
model_type: "lgbm_gp"

features:
  - stage_id
  - cumu_gdd_air
  - rolling_gdd_air
  - rolling_humidity_mean_pct
  - cumu_precip_total_mm
  - latitude
  - longitude

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
  monotone_constraints_by_feature:
    cumu_gdd_air: -1
```

### Config Rules

- `monotone_constraints_by_feature` is the human-authored source of truth
- `features` remains the source of feature order
- Any feature not listed in `monotone_constraints_by_feature` defaults to `0`
- The final ordered list passed to LightGBM must follow `features` exactly

For the current feature list, the generated LightGBM parameter must be:

```yaml
[0, -1, 0, 0, 0, 0, 0]
```

## Validation Rules

The config-to-parameter conversion should fail fast.

- Every key in `monotone_constraints_by_feature` must exist in `config.features`
- Every constraint value must be one of `-1`, `0`, `1`
- The generated `monotone_constraints` list must have the same length as `config.features`

Error handling should be strict:

- Unknown feature name -> raise an error
- Invalid constraint value -> raise an error
- Silent ignore is not allowed

## Implementation Design

### Responsibility Split

- Config layer:
  - declares `monotone_constraints_by_feature`
- Model manager layer:
  - validates the dict
  - converts it to the ordered LightGBM `monotone_constraints` list
  - passes only LightGBM-native params to the estimator
- Estimator layer:
  - receives the final LightGBM-ready parameters
  - uses them during `fit()`

### Why Parse in the Model Manager

The manager is the best place to normalize parameters because:

- it already owns model-specific training params
- it can validate against `config.features`
- it keeps the estimator interface closer to LightGBM's native API

The estimator should not need to understand the custom `*_by_feature` config format.

## Model-Type Behavior

Monotonic constraints are only applied on the LightGBM path.

For non-LightGBM models such as RF:

- the config may still contain `monotone_constraints_by_feature`
- the RF path should ignore it
- optional logging is acceptable to indicate that the current model type does not use monotonic constraints

The RF path should not fail just because the field exists.

## Minimal Verification for V1

V1 does not require backtesting.

The minimum verification target is:

1. Confirm config parsing works
2. Confirm LightGBM training still succeeds
3. Confirm local monotonicity is actually enforced

### Simplest Local Monotonicity Check

Use one trained LightGBM model and one base sample:

1. Pick or construct a valid input row
2. Duplicate it multiple times
3. Change only `cumu_gdd_air`
4. Increase `cumu_gdd_air` stepwise, for example:
   - `100`
   - `150`
   - `200`
   - `250`
5. Keep all other features identical
6. Predict on the modified rows
7. Verify predictions are non-increasing

Expected property:

```text
pred(100) >= pred(150) >= pred(200) >= pred(250)
```

This is sufficient for V1 because it verifies the exact structural behavior we are introducing.

## Acceptance Criteria

V1 is complete when all of the following are true:

- A LightGBM config can declare `model_parameters.monotone_constraints_by_feature`
- The code converts that declaration into an ordered LightGBM `monotone_constraints` list using `config.features`
- V1 supports `cumu_gdd_air: -1`
- Invalid config fails fast with clear errors
- LightGBM training and prediction still run successfully
- A minimal local monotonicity check confirms predictions are non-increasing as `cumu_gdd_air` increases

## Deferred Items

The following are intentionally postponed to later phases:

- adding `cumu_precip_total_mm: -1`
- evaluating whether extra constraints improve or hurt accuracy
- GP residual correction
- prediction intervals
- zero-day termination
- backtesting-based evaluation

## Notes for Future Expansion

If V1 behaves well, the next step can extend the same config mechanism to additional constrained features. The first candidate to evaluate later is:

- `cumu_precip_total_mm: -1`

That should remain an empirical decision, not a default assumption in V1.

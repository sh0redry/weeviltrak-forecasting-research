"""
Learned termination post-processing for WeevilTrak v2.4.

The primary model remains the configured LightGBM signed-days predictor.  This
module trains and applies a small Ridge calibrator that decides when a
trajectory should terminate based only on prediction history visible up to the
current prediction date.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:  # Spline crossings are useful, but v2.4 can still run without SciPy.
    from scipy.interpolate import UnivariateSpline

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - depends on optional runtime dependency
    UnivariateSpline = None
    SCIPY_AVAILABLE = False


TERMINATION_NUMERIC_FEATURES = [
    "prediction_day_of_year",
    "feature_latitude",
    "feature_longitude",
    "termination_signal_days",
    "raw_signed_days",
    "predicted_days",
    "signal_lag1",
    "signal_lag2",
    "signal_lag3",
    "slope_1",
    "slope_2",
    "slope_3",
    "recent_mean_3",
    "recent_std_3",
    "recent_min_3",
    "recent_max_3",
    "rolling_linear_crossing_0",
    "rolling_linear_crossing_minus_0_5",
    "rolling_spline_crossing_0",
    "rolling_spline_crossing_minus_0_5",
    "days_to_rolling_crossing_0",
    "days_to_rolling_crossing_minus_0_5",
]
TERMINATION_CATEGORICAL_FEATURES = ["stage"]
TERMINATION_FEATURES = TERMINATION_CATEGORICAL_FEATURES + TERMINATION_NUMERIC_FEATURES


@dataclass(frozen=True)
class LearnedTerminationSettings:
    enabled: bool = False
    alpha: float = 30.0
    threshold: float = 4.5
    model_path: Optional[str] = None
    overwrite_business_prediction: bool = True
    min_history_days: int = 0
    training_window_years: Optional[int] = None

    @classmethod
    def from_config(cls, config: dict) -> "LearnedTerminationSettings":
        raw = config.get("termination_model", {}) or {}
        training_window_years = raw.get("training_window_years")
        min_history_days = raw.get("min_history_days", 0)
        if (
            isinstance(min_history_days, bool)
            or not isinstance(min_history_days, int)
            or min_history_days < 0
        ):
            raise ValueError(
                "termination_model.min_history_days must be a non-negative int, "
                f"got {min_history_days!r}"
            )
        if training_window_years is not None:
            if (
                isinstance(training_window_years, bool)
                or not isinstance(training_window_years, int)
                or training_window_years <= 0
            ):
                raise ValueError(
                    "termination_model.training_window_years must be a positive "
                    f"int or None, got {training_window_years!r}"
                )
        return cls(
            enabled=bool(raw.get("enabled", False)),
            alpha=float(raw.get("alpha", 30.0)),
            threshold=float(raw.get("threshold", 4.5)),
            model_path=raw.get("model_path"),
            overwrite_business_prediction=bool(
                raw.get("overwrite_business_prediction", True)
            ),
            min_history_days=min_history_days,
            training_window_years=training_window_years,
        )


def select_aligned_training_window(
    features: pd.DataFrame,
    target: pd.DataFrame,
    cutoff_date,
    window_years: Optional[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a date window to aligned feature and target rows."""
    if len(features) != len(target):
        raise ValueError("features and target must contain the same number of rows")
    if window_years is None:
        return features, target
    if (
        isinstance(window_years, bool)
        or not isinstance(window_years, int)
        or window_years <= 0
    ):
        raise ValueError("window_years must be a positive int or None")

    dates = pd.to_datetime(features.index, errors="coerce")
    if dates.isna().any():
        raise ValueError("training feature index must contain valid dates")
    cutoff = pd.to_datetime(cutoff_date)
    start = cutoff - pd.DateOffset(years=window_years)
    positions = np.flatnonzero((dates >= start) & (dates <= cutoff))
    if len(positions) == 0:
        raise ValueError(
            f"No termination training rows in window {start.date()} to {cutoff.date()}"
        )
    return features.iloc[positions].copy(), target.iloc[positions].copy()


def _all_crossings(x: np.ndarray, y: np.ndarray, threshold: float) -> list[float]:
    crossings: list[float] = []
    diff = y - threshold
    for i in range(1, len(x)):
        if not np.isfinite(diff[i - 1]) or not np.isfinite(diff[i]):
            continue
        if np.isclose(diff[i - 1], 0.0):
            crossings.append(float(x[i - 1]))
        elif np.isclose(diff[i], 0.0):
            crossings.append(float(x[i]))
        elif (diff[i - 1] > 0 and diff[i] < 0) or (diff[i - 1] < 0 and diff[i] > 0):
            ratio = diff[i - 1] / (diff[i - 1] - diff[i])
            crossings.append(float(x[i - 1] + ratio * (x[i] - x[i - 1])))
    return crossings


def _choose_crossing_for_current(crossings: list[float], current_doy: float) -> float:
    if not crossings:
        return np.nan
    future = [value for value in crossings if value >= current_doy]
    if future:
        return float(min(future))
    return float(max(crossings))


def _rolling_linear_crossing(x: np.ndarray, y: np.ndarray, threshold: float) -> float:
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(np.unique(x)) < 2:
        return np.nan
    try:
        slope, intercept = np.polyfit(x, y, 1)
    except Exception:
        return np.nan
    if np.isclose(slope, 0.0):
        return np.nan
    crossing = (threshold - intercept) / slope
    return float(crossing) if np.isfinite(crossing) else np.nan


def _rolling_spline_crossing(
    x: np.ndarray, y: np.ndarray, threshold: float, current_doy: float
) -> float:
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if not SCIPY_AVAILABLE or len(np.unique(x)) < 6:
        return np.nan
    try:
        order = np.argsort(x)
        x, y = x[order], y[order]
        if len(x) > 12:
            x, y = x[-12:], y[-12:]
        k = min(3, len(np.unique(x)) - 1)
        smoothing = max(len(x) * np.nanvar(y) * 0.35, 1e-6)
        # FITPACK can return a usable approximation while warning that its
        # smoothing tolerance was not met. Keep that established behavior
        # without flooding batch backtests with one warning per trajectory.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            spline = UnivariateSpline(x, y, s=smoothing, k=k)
        end_doy = min(max(float(np.nanmax(x)), current_doy + 45.0), 210.0)
        grid = np.linspace(float(np.nanmin(x)), end_doy, 120)
        values = spline(grid)
        return _choose_crossing_for_current(
            _all_crossings(grid, values, threshold), current_doy
        )
    except Exception:
        return np.nan


def _rolling_crossing_features_for_group(
    group: pd.DataFrame, spline_update_every: int = 3
) -> pd.DataFrame:
    group = group.sort_values("prediction_date").copy()
    x_all = group["prediction_day_of_year"].to_numpy(dtype=float)
    y_all = group["signal"].to_numpy(dtype=float)
    rows = []
    for i in range(len(group)):
        current_doy = float(x_all[i])
        x_hist = x_all[: i + 1]
        y_hist = y_all[: i + 1]
        linear_0 = _rolling_linear_crossing(x_hist, y_hist, 0.0)
        linear_m05 = _rolling_linear_crossing(x_hist, y_hist, -0.5)
        should_update_spline = i >= 5 and (
            i % spline_update_every == 0 or i == len(group) - 1
        )
        spline_0 = (
            _rolling_spline_crossing(x_hist, y_hist, 0.0, current_doy)
            if should_update_spline
            else np.nan
        )
        spline_m05 = (
            _rolling_spline_crossing(x_hist, y_hist, -0.5, current_doy)
            if should_update_spline
            else np.nan
        )
        rows.append(
            {
                "rolling_linear_crossing_0": linear_0,
                "rolling_linear_crossing_minus_0_5": linear_m05,
                "rolling_spline_crossing_0": spline_0,
                "rolling_spline_crossing_minus_0_5": spline_m05,
            }
        )

    features = pd.DataFrame(rows, index=group.index)
    features["rolling_spline_crossing_0"] = features[
        "rolling_spline_crossing_0"
    ].ffill()
    features["rolling_spline_crossing_minus_0_5"] = features[
        "rolling_spline_crossing_minus_0_5"
    ].ffill()
    crossing_0 = features["rolling_spline_crossing_0"].where(
        features["rolling_spline_crossing_0"].notna(),
        features["rolling_linear_crossing_0"],
    )
    crossing_m05 = features["rolling_spline_crossing_minus_0_5"].where(
        features["rolling_spline_crossing_minus_0_5"].notna(),
        features["rolling_linear_crossing_minus_0_5"],
    )
    features["days_to_rolling_crossing_0"] = crossing_0 - group[
        "prediction_day_of_year"
    ].astype(float)
    features["days_to_rolling_crossing_minus_0_5"] = crossing_m05 - group[
        "prediction_day_of_year"
    ].astype(float)
    return features


def _group_columns(df: pd.DataFrame) -> list[str]:
    # A trajectory is seasonal.  Omitting the year would incorrectly make the
    # final prediction of one season a lag for the next season.
    season_col = next(
        (
            col
            for col in ("trajectory_year", "test_year", "prediction_year")
            if col in df.columns and df[col].notna().any()
        ),
        None,
    )
    if "location_id" in df.columns and df["location_id"].notna().any():
        cols = ["location_id", "stage_id"]
    else:
        cols = ["feature_latitude", "feature_longitude", "stage_id"]
    return ([season_col] if season_col else []) + cols


def build_termination_feature_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Build v2.4 learned termination features from visible trajectory history.
    """
    if predictions is None or predictions.empty:
        return pd.DataFrame()

    original_index = predictions.index
    out = predictions.copy()
    if "prediction_date" not in out.columns and isinstance(out.index, pd.DatetimeIndex):
        out["prediction_date"] = out.index
    out = out.reset_index(drop=True)
    if "prediction_date" not in out.columns:
        if isinstance(original_index, pd.DatetimeIndex):
            out["prediction_date"] = original_index
        else:
            raise ValueError("prediction_date is required for termination features")

    out["prediction_date"] = pd.to_datetime(out["prediction_date"], errors="coerce")
    out["stage_id"] = pd.to_numeric(out["stage_id"], errors="coerce").astype(int)
    out["raw_signed_days"] = pd.to_numeric(out["raw_signed_days"], errors="coerce")
    out["predicted_days"] = pd.to_numeric(out["predicted_days"], errors="coerce")
    out["termination_signal_days"] = out["raw_signed_days"].where(
        out["raw_signed_days"].notna(), out["predicted_days"]
    )
    out["signal"] = out["termination_signal_days"]
    out["prediction_day_of_year"] = out["prediction_date"].dt.dayofyear.astype(float)
    out["feature_latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["feature_longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out["stage"] = out["stage_id"].astype(str)

    if "location_id" in out.columns:
        out["location_id"] = out["location_id"].astype(str)

    sort_cols = _group_columns(out) + ["prediction_date"]
    out = out.sort_values(sort_cols).copy()
    group_cols = _group_columns(out)
    grouped = out.groupby(group_cols, sort=False)

    for lag in [1, 2, 3]:
        out[f"signal_lag{lag}"] = grouped["signal"].shift(lag)

    out["slope_1"] = out["signal"] - out["signal_lag1"]
    out["slope_2"] = out["signal_lag1"] - out["signal_lag2"]
    out["slope_3"] = out["signal_lag2"] - out["signal_lag3"]
    out["recent_mean_3"] = grouped["signal"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    out["recent_std_3"] = grouped["signal"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=2).std()
    )
    out["recent_min_3"] = grouped["signal"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).min()
    )
    out["recent_max_3"] = grouped["signal"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).max()
    )

    crossing_parts = [
        _rolling_crossing_features_for_group(group)
        for _, group in out.groupby(group_cols, sort=False)
    ]
    if crossing_parts:
        out = pd.concat([out, pd.concat(crossing_parts).sort_index()], axis=1)

    out = out.sort_index()
    out.index = original_index
    return out


def make_learned_termination_model(alpha: float = 30.0) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                TERMINATION_NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                TERMINATION_CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("model", Ridge(alpha=alpha))])


def train_learned_termination_model(
    training_frame: pd.DataFrame,
    target_days_remaining: Iterable[float],
    alpha: float = 30.0,
) -> Pipeline:
    features = build_termination_feature_frame(training_frame)
    if features.empty:
        raise ValueError("Cannot train learned termination model from empty data")
    target = pd.to_numeric(
        pd.Series(
            np.asarray(list(target_days_remaining), dtype=float), index=features.index
        ),
        errors="coerce",
    )
    valid = target.notna()
    model = make_learned_termination_model(alpha=alpha)
    model.fit(
        features.loc[valid, TERMINATION_FEATURES], target.loc[valid].astype(float)
    )
    return model


def apply_learned_termination_rule(
    predictions: pd.DataFrame,
    model: Pipeline,
    threshold: float = 4.5,
    overwrite_business_prediction: bool = True,
    min_history_days: int = 0,
) -> pd.DataFrame:
    """
    Add learned termination columns and optionally overwrite business output.

    ``learned_days_remaining`` is kept as the raw Ridge latent signal.  The
    business-facing remaining-days curve is calibrated by shifting that signal
    down by the operating threshold and clipping at zero.
    """
    if predictions is None or predictions.empty:
        return predictions
    if (
        isinstance(min_history_days, bool)
        or not isinstance(min_history_days, int)
        or min_history_days < 0
    ):
        raise ValueError("min_history_days must be a non-negative int")

    # Daily production inputs are commonly indexed by prediction date.  That
    # index is necessarily duplicated across locations and Stage/Phase output
    # rows, while the learned-feature frame uses a per-row alignment.  Keep
    # ``prediction_date`` as the business date column and give every row a
    # stable unique index before building/aligning termination features.
    #
    # This belongs here rather than at individual callers so train/predict,
    # backtesting, hindcast, and external deployment clients have identical
    # behavior for a valid multi-location daily trajectory.
    out = predictions.copy()
    if "prediction_date" not in out.columns and isinstance(out.index, pd.DatetimeIndex):
        out["prediction_date"] = out.index
    out = out.reset_index(drop=True)
    features = build_termination_feature_frame(out)
    learned_days = model.predict(features[TERMINATION_FEATURES])
    aligned = pd.Series(learned_days, index=features.index).reindex(out.index)

    out["termination_rule"] = "learned_ridge_v2.4"
    out["termination_threshold"] = float(threshold)
    out["learned_days_remaining"] = aligned.astype(float)
    out["termination_adjusted_days_remaining"] = (
        out["learned_days_remaining"] - float(threshold)
    ).clip(lower=0.0)
    out["termination_first_hit"] = False
    out["termination_reached"] = False
    out["termination_history_days"] = 0
    out["termination_output_applied"] = False
    out["termination_decision_date"] = pd.NaT

    prediction_dates = pd.to_datetime(out["prediction_date"], errors="coerce")
    work = out.assign(_prediction_date=prediction_dates).sort_values(
        _group_columns(out) + ["_prediction_date"]
    )
    for _, group in work.groupby(_group_columns(work), sort=False, dropna=False):
        visible_dates = pd.Index(group["_prediction_date"].drop_duplicates())
        history_days = pd.Series(
            np.arange(1, len(visible_dates) + 1), index=visible_dates
        )
        group_history_days = group["_prediction_date"].map(history_days).astype(int)
        out.loc[group.index, "termination_history_days"] = group_history_days
        # ``min_history_days=7`` means the first seven daily outputs remain
        # provisional Base predictions; learned termination can first govern
        # the eighth visible prediction day.
        eligible_indexes = group.index[group_history_days.gt(min_history_days)]
        out.loc[eligible_indexes, "termination_output_applied"] = True
        hits = group.loc[eligible_indexes]
        hits = hits[hits["termination_adjusted_days_remaining"].le(0.0)]
        if hits.empty:
            continue
        first_index = hits.index[0]
        decision_date = prediction_dates.loc[first_index]
        lock_indexes = group.loc[group["_prediction_date"].ge(decision_date)].index
        out.loc[first_index, "termination_first_hit"] = True
        out.loc[lock_indexes, "termination_reached"] = True
        out.loc[lock_indexes, "termination_decision_date"] = decision_date
        # Whitepaper business contract: after the first termination decision,
        # remain at zero instead of allowing a later bounce-up.
        out.loc[lock_indexes, "termination_adjusted_days_remaining"] = 0.0

    business_stage_dates = prediction_dates + pd.to_timedelta(
        out["termination_adjusted_days_remaining"], unit="D"
    )
    decision_dates = pd.to_datetime(out["termination_decision_date"], errors="coerce")
    business_stage_dates = business_stage_dates.where(
        decision_dates.isna(), decision_dates
    )
    out["termination_estimated_stage_date"] = business_stage_dates

    if overwrite_business_prediction:
        if "base_predicted_days" not in out.columns:
            out["base_predicted_days"] = out["predicted_days"]
        if "base_predicted_stage_date" not in out.columns:
            out["base_predicted_stage_date"] = out["predicted_stage_date"]
        adjusted_days = out["termination_adjusted_days_remaining"]
        has_adjusted_days = adjusted_days.notna() & out["termination_output_applied"]
        out.loc[has_adjusted_days, "predicted_days"] = adjusted_days.loc[
            has_adjusted_days
        ].astype(float)
        out.loc[has_adjusted_days, "predicted_stage_date"] = business_stage_dates.loc[
            has_adjusted_days
        ].dt.strftime("%Y-%m-%d")

    return out

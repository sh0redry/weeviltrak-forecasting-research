"""Generate reproducible v2.4 Whitepaper feature-ablation tables.

The base-layer experiment refits LightGBM for each held-out year (2023--25)
using the production three-year window.  The termination experiment performs
the analogous chronological OOF Ridge evaluation from cached v2.4 trajectories.
Neither experiment uses the final production Ridge artifact.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.models.factory import create_model_manager
from app.pipeline.backtesting import BacktestingFramework
from app.pipeline.termination import TERMINATION_FEATURES, build_termination_feature_frame


YEARS = (2023, 2024, 2025)
STAGES = (1, 2, 3)
BASE_GROUPS = {
    "lifecycle context": ("stage_id",),
    "heat accumulation": ("cumu_gdd_air", "rolling_gdd_air"),
    "moisture": ("rolling_humidity_mean_pct", "cumu_precip_total_mm"),
    "cold accumulation": ("cumu_cdd_air", "rolling_cdd_air"),
    "seasonal and spatial context": ("doy", "latitude", "longitude"),
}
TERMINATION_GROUPS = {
    "static context": ("stage", "prediction_day_of_year", "feature_latitude", "feature_longitude"),
    "base prediction signal": ("termination_signal_days", "raw_signed_days", "predicted_days"),
    "recent trajectory shape": ("signal_lag1", "signal_lag2", "signal_lag3", "slope_1", "slope_2", "slope_3", "recent_mean_3", "recent_std_3", "recent_min_3", "recent_max_3"),
    "estimated crossing": ("rolling_linear_crossing_0", "rolling_linear_crossing_minus_0_5", "rolling_spline_crossing_0", "rolling_spline_crossing_minus_0_5", "days_to_rolling_crossing_0", "days_to_rolling_crossing_minus_0_5"),
}


def _metrics(errors: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(errors, errors="coerce").dropna().astype(float)
    return {
        "n": int(len(values)),
        "mae_days": float(values.abs().mean()),
        "rmse_days": float(np.sqrt(np.mean(np.square(values)))),
    }


def _ablation_config(config, features: Iterable[str]):
    cfg = copy.deepcopy(config)
    selected = list(features)
    cfg.features = selected
    cfg.config["features"] = selected
    params = copy.deepcopy(cfg.config.get("model_parameters", {}))
    constraints = params.get("monotone_constraints_by_feature", {})
    params["monotone_constraints_by_feature"] = {
        name: value for name, value in constraints.items() if name in selected
    }
    cfg.model_params = params
    cfg.config["model_parameters"] = params
    return cfg


def _base_predictions(backtester, weather, detail, features) -> pd.DataFrame:
    """Refit a base model per year, predict exactly the cached evaluation keys."""
    predictions: list[pd.DataFrame] = []
    for year in YEARS:
        scope = detail[(detail.test_year == year) & detail.stage_id.isin(STAGES)].copy()
        if scope.empty:
            continue
        cfg = _ablation_config(backtester.config, features)
        manager = create_model_manager(cfg, backtester.s3_manager)
        cutoff = f"{year}-01-01"
        weevil = backtester.data_service.pull_weevil_data(today=cutoff)
        x_train, y_train = backtester.data_service.prepare_training_data(
            weevil_data=weevil,
            weather_data=weather,
            test_date=cutoff,
            features=list(features),
            target=cfg.target,
            window_years=3,
            post_event_training_days=int(cfg.config.get("post_event_training_days", 10)),
        )
        manager.train(x_train=x_train, y_train=y_train)
        for prediction_date, points in scope.groupby("prediction_date", sort=False):
            date = pd.to_datetime(prediction_date).strftime("%Y-%m-%d")
            # ``stage_id`` remains an evaluation key even for the ablation that
            # deliberately removes it from the model input.
            test_columns = list(features)
            if "stage_id" not in test_columns:
                test_columns.append("stage_id")
            x = backtester.data_service.prepare_test_data(weather, test_columns, date)
            if x.empty:
                continue
            key = points[["location_id", "stage_id", "actual_stage_date"]].copy()
            x = x.reset_index(drop=False).rename(columns={x.index.name or "index": "prediction_date"})
            x["prediction_date"] = pd.to_datetime(x["prediction_date"]).dt.strftime("%Y-%m-%d")
            x["location_id"] = x["location_id"].astype(str)
            key["location_id"] = key["location_id"].astype(str)
            key["prediction_date"] = date
            x = x.merge(key, on=["prediction_date", "location_id", "stage_id"], how="inner")
            if x.empty:
                continue
            raw = np.asarray(manager.predict(x[list(features)]), dtype=float)
            x["raw_predicted_stage_date"] = pd.to_datetime(date) + pd.to_timedelta(raw, unit="D")
            x["error_days"] = (x["raw_predicted_stage_date"] - pd.to_datetime(x["actual_stage_date"])).dt.total_seconds() / 86400
            predictions.append(x[["prediction_date", "location_id", "stage_id", "actual_stage_date", "error_days"]])
    if not predictions:
        raise RuntimeError("No base ablation predictions were generated.")
    return pd.concat(predictions, ignore_index=True)


def generate_base_ablations(backtester, weather, detail, out_dir: Path) -> None:
    full_features = list(backtester.config.features)
    baseline = _base_predictions(backtester, weather, detail, full_features)
    baseline.to_parquet(out_dir / "base_full_predictions.parquet", index=False)
    baseline_metrics = _metrics(baseline.error_days)

    def evaluate(specs, name_col, filename):
        rows = []
        for name, removed in specs.items():
            selected = [feature for feature in full_features if feature not in removed]
            result = _base_predictions(backtester, weather, detail, selected)
            merged = baseline.merge(result, on=["prediction_date", "location_id", "stage_id", "actual_stage_date"], suffixes=("_full", "_ablated"))
            full = _metrics(merged.error_days_full)
            ablated = _metrics(merged.error_days_ablated)
            rows.append({
                name_col: name, "removed_features": ", ".join(removed), "n_matched": ablated["n"],
                "full_mae_days": full["mae_days"], "ablated_mae_days": ablated["mae_days"],
                "delta_mae": ablated["mae_days"] - full["mae_days"],
                "full_rmse_days": full["rmse_days"], "ablated_rmse_days": ablated["rmse_days"],
                "delta_rmse": ablated["rmse_days"] - full["rmse_days"],
            })
        pd.DataFrame(rows).sort_values("delta_rmse", ascending=False).to_csv(out_dir / filename, index=False)

    evaluate(BASE_GROUPS, "feature_group", "base_feature_group_ablation.csv")
    evaluate({feature: (feature,) for feature in full_features}, "feature", "base_individual_feature_ablation.csv")
    pd.DataFrame([baseline_metrics]).to_csv(out_dir / "base_full_metrics.csv", index=False)


def _ridge(features: list[str], alpha: float) -> Pipeline:
    categorical = [col for col in features if col == "stage"]
    numeric = [col for col in features if col != "stage"]
    steps = []
    if numeric:
        steps.append(("num", Pipeline([( "imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric))
    if categorical:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    return Pipeline([("preprocessor", ColumnTransformer(steps)), ("model", Ridge(alpha=alpha))])


def _apply_ridge(frame: pd.DataFrame, model, features: list[str], threshold: float) -> pd.DataFrame:
    out = frame.copy()
    out["learned"] = model.predict(out[features])
    out["adjusted"] = (out.learned - threshold).clip(lower=0)
    out["decision_date"] = pd.NaT
    group_cols = ["test_year", "location_id", "stage_id"]
    for _, group in out.sort_values("prediction_date").groupby(group_cols, sort=False):
        hit = group[group.adjusted.le(0)]
        if not hit.empty:
            out.loc[group.index, "decision_date"] = hit.iloc[0].prediction_date
    events = out.dropna(subset=["decision_date", "actual_stage_date"]).copy()
    events = events.sort_values("prediction_date").groupby(group_cols, as_index=False).first()
    events["error_days"] = (pd.to_datetime(events.decision_date) - pd.to_datetime(events.actual_stage_date)).dt.total_seconds() / 86400
    return events


def generate_termination_ablation(detail: pd.DataFrame, out_dir: Path, alpha: float = 30.0, threshold: float = 4.5) -> None:
    source = detail.copy()
    source["prediction_date"] = pd.to_datetime(source["prediction_date"])
    source["actual_stage_date"] = pd.to_datetime(source["actual_stage_date"])
    # Termination features must originate from the base trajectory, not final output.
    if "base_predicted_days" in source:
        source["predicted_days"] = source["base_predicted_days"]
    frame = build_termination_feature_frame(source)
    frame["test_year"] = source["test_year"].to_numpy()
    frame["actual_stage_date"] = source["actual_stage_date"].to_numpy()
    frame["prediction_date"] = source["prediction_date"].to_numpy()
    frame = frame[frame.stage_id.isin(STAGES) & frame.test_year.isin((2022,) + YEARS)].copy()
    frame["target_days"] = (frame.actual_stage_date - frame.prediction_date).dt.total_seconds() / 86400

    def evaluate(features: list[str]) -> pd.DataFrame:
        events = []
        for year in YEARS:
            train = frame[(frame.test_year < year) & frame.target_days.notna()]
            test = frame[frame.test_year.eq(year)].copy()
            model = _ridge(features, alpha)
            model.fit(train[features], train.target_days)
            events.append(_apply_ridge(test, model, features, threshold))
        return pd.concat(events, ignore_index=True)

    baseline = evaluate(list(TERMINATION_FEATURES))
    baseline.to_parquet(out_dir / "termination_full_events.parquet", index=False)
    def evaluate_specs(specs: dict[str, tuple[str, ...]], name_col: str, filename: str) -> None:
        rows = []
        for name, removed in specs.items():
            features = [feature for feature in TERMINATION_FEATURES if feature not in removed]
            ablated = evaluate(features)
            keys = ["test_year", "location_id", "stage_id", "actual_stage_date"]
            merged = baseline.merge(ablated, on=keys, suffixes=("_full", "_ablated"))
            full, reduced = _metrics(merged.error_days_full), _metrics(merged.error_days_ablated)
            rows.append({name_col: name, "removed_features": ", ".join(removed), "n_matched": reduced["n"], "full_mae_days": full["mae_days"], "ablated_mae_days": reduced["mae_days"], "delta_mae": reduced["mae_days"] - full["mae_days"], "full_rmse_days": full["rmse_days"], "ablated_rmse_days": reduced["rmse_days"], "delta_rmse": reduced["rmse_days"] - full["rmse_days"]})
        pd.DataFrame(rows).sort_values("delta_rmse", ascending=False).to_csv(out_dir / filename, index=False)

    evaluate_specs(TERMINATION_GROUPS, "feature_group", "termination_feature_group_ablation.csv")
    evaluate_specs(
        {feature: (feature,) for feature in TERMINATION_FEATURES},
        "feature",
        "termination_individual_feature_ablation.csv",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default="app/config/weeviltrak_v2.4.yml")
    parser.add_argument("--cache-dir", default="outputs/backtesting/paired_protocol/v24_window3")
    parser.add_argument("--output-dir", default="outputs/report_figures/v2_4/controlled_oof/ablation")
    parser.add_argument("--termination-only", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir)
    detail = pd.read_parquet(cache / "trajectory_detail.parquet")
    if "test_year" not in detail:
        detail["test_year"] = pd.to_datetime(detail["prediction_date"]).dt.year
    if not args.termination_only:
        backtester = BacktestingFramework(args.config_path, output_dir=str(out_dir / "_scratch"))
        weather = pd.read_parquet(cache / "processed_weather_unique.parquet")
        generate_base_ablations(backtester, weather, detail, out_dir)
    generate_termination_ablation(detail, out_dir)


if __name__ == "__main__":
    main()

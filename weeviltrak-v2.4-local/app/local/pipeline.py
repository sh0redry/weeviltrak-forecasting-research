"""Reproducible local WeevilTrak v2.4 research workflow.

The workflow keeps the v2.4 feature schema, signed-days target, LightGBM
parameters, canonical Stage/Phase IDs, three-year window, chronological
walk-forward split, and learned termination implementation.  It accepts an
engineered daily-feature export.  A deterministic synthetic generator exists
only for smoke testing and must not be used for research conclusions.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from app.config.weevilltrak_config import WeevillTrakConfig
from app.models.factory import create_model_manager
from app.pipeline.termination import (
    LearnedTerminationSettings,
    apply_learned_termination_rule,
    train_learned_termination_model,
)
from app.services.canonical_events import (
    CanonicalEventConfig,
    add_stage_output_names,
    append_phase1_midpoint_training_events,
)
from app.services.manual_events import ManualEventConfig, load_manual_stage_events
from app.services.training_coverage import (
    RecentLabelCoveragePolicy,
    require_recent_label_coverage,
)
from griddedweather.s3_store import S3Manager


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "app/config/weeviltrak_v2.4_local.yml"
DEFAULT_EVENTS = (
    PROJECT_ROOT
    / "data_exports/v24_redshift_training_events_2026-08-21"
    / "v24_canonical_training_event_labels_2023-08-21_to_2026-08-21.csv"
)
DEFAULT_FEATURES = PROJECT_ROOT / "data/local/mock_engineered_daily_features.csv"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "outputs/local/artifacts"
DEFAULT_BACKTEST_DIR = PROJECT_ROOT / "outputs/backtesting/v2.4-local"

IDENTITY_COLUMNS = ["date", "location_id", "place_id", "latitude", "longitude"]


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_events(path: str | Path = DEFAULT_EVENTS) -> pd.DataFrame:
    source = _resolve(path)
    frame = pd.read_csv(source)
    required = {"location_id", "latitude", "longitude", "stage_id", "stage_date"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Event export is missing columns: {missing}")
    frame["stage_date"] = pd.to_datetime(frame["stage_date"], errors="coerce")
    frame["location_id"] = frame["location_id"].astype(str)
    frame["stage_id"] = pd.to_numeric(frame["stage_id"], errors="coerce")
    frame = frame.dropna(subset=["stage_date", "stage_id", "latitude", "longitude"])
    frame["stage_id"] = frame["stage_id"].astype(int)
    frame = frame[frame["stage_id"].isin([1, 4, 2, 3])].copy()
    frame["event_year"] = frame["stage_date"].dt.year
    return frame.sort_values(["stage_date", "location_id", "stage_id"])


def add_configured_manual_events(
    events: pd.DataFrame, config: WeevillTrakConfig
) -> pd.DataFrame:
    """Add the curated runtime events without duplicating exported Phase 1 rows."""
    manual_settings = ManualEventConfig.from_config(config.config)
    if not manual_settings.enabled or not manual_settings.include_in_training:
        return events
    manual = load_manual_stage_events()
    canonical = CanonicalEventConfig.from_config(config.config)
    manual = append_phase1_midpoint_training_events(manual, config=canonical)
    combined = pd.concat([events, manual], ignore_index=True, sort=False)
    combined["stage_date"] = pd.to_datetime(combined["stage_date"], errors="coerce")
    combined["location_id"] = combined["location_id"].astype(str)
    combined["stage_id"] = pd.to_numeric(combined["stage_id"], errors="coerce").astype(int)
    combined["event_year"] = combined["stage_date"].dt.year
    return combined.drop_duplicates(
        ["location_id", "event_year", "stage_id"], keep="first"
    ).sort_values(["stage_date", "location_id", "stage_id"])


def generate_mock_engineered_features(
    events: pd.DataFrame,
    output_path: str | Path = DEFAULT_FEATURES,
) -> Path:
    """Create deterministic, schema-valid synthetic daily features.

    These values are not observed weather.  The sidecar metadata makes that
    boundary machine-readable as well as visible in the documentation.
    """
    locations = events[
        ["location_id", "event_year", "latitude", "longitude"]
    ].drop_duplicates()
    rows: list[pd.DataFrame] = []
    for item in locations.itertuples(index=False):
        dates = pd.date_range(f"{item.event_year}-03-01", f"{item.event_year}-06-30")
        doy = dates.dayofyear.to_numpy(dtype=float)
        phase = (float(item.latitude) * 13.0 + float(item.longitude) * 7.0) % 9.0
        mean_c = 5.5 + 0.19 * (doy - 60.0) + 1.8 * np.sin((doy + phase) / 8.0)
        mean_f = mean_c * 9.0 / 5.0 + 32.0
        daily_gdd = np.clip(mean_f - 50.0, 0.0, None)
        daily_cdd = np.clip(50.0 - mean_f, 0.0, None)
        humidity = 66.0 + 10.0 * np.sin((doy + phase) / 6.0)
        precip = np.where(((doy.astype(int) + int(phase)) % 9) == 0, 5.0, 0.4)
        local = pd.DataFrame(
            {
                "date": dates,
                "location_id": str(item.location_id),
                "place_id": f"local-{item.location_id}",
                "latitude": float(item.latitude),
                "longitude": float(item.longitude),
                "gdd_air": daily_gdd,
                "cdd_air": daily_cdd,
                "humidity_mean_pct": humidity,
                "precip_total_mm": precip,
            }
        )
        local["cumu_gdd_air"] = local["gdd_air"].cumsum()
        local["cumu_cdd_air"] = local["cdd_air"].cumsum()
        local["cumu_precip_total_mm"] = local["precip_total_mm"].cumsum()
        local["rolling_gdd_air"] = local["gdd_air"].rolling(14, min_periods=1).mean()
        local["rolling_cdd_air"] = local["cdd_air"].rolling(14, min_periods=1).mean()
        local["rolling_humidity_mean_pct"] = local["humidity_mean_pct"].rolling(
            14, min_periods=1
        ).mean()
        local["doy"] = local["date"].dt.dayofyear
        rows.append(local)
    output = _resolve(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(output, index=False)
    metadata = {
        "kind": "synthetic_smoke_test_only",
        "deterministic": True,
        "research_use_allowed": False,
        "source_event_rows": int(len(events)),
        "generator": "app.local.pipeline.generate_mock_engineered_features",
    }
    output.with_suffix(output.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return output


def load_engineered_features(
    path: str | Path,
    configured_features: Iterable[str],
) -> pd.DataFrame:
    source = _resolve(path)
    if source.suffix == ".parquet":
        frame = pd.read_parquet(source)
    else:
        frame = pd.read_csv(source)
    required = set(IDENTITY_COLUMNS) | (set(configured_features) - {"stage_id"})
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Engineered feature file is missing columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["location_id"] = frame["location_id"].astype(str)
    numeric = sorted(required - {"date", "location_id", "place_id"})
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["date", *numeric]].isna().any().any():
        bad = frame[["date", *numeric]].columns[
            frame[["date", *numeric]].isna().any()
        ].tolist()
        raise ValueError(f"Engineered features contain missing/non-numeric values: {bad}")
    duplicate = frame.duplicated(["date", "location_id"])
    if duplicate.any():
        raise ValueError(
            f"Engineered features contain {int(duplicate.sum())} duplicate date/location rows"
        )
    return frame.sort_values(["location_id", "date"]).reset_index(drop=True)


def build_training_matrix(
    events: pd.DataFrame,
    daily: pd.DataFrame,
    config: WeevillTrakConfig,
    cutoff_date: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(cutoff_date).normalize()
    window_years = int(config.config.get("training_window_years", 3))
    start = cutoff - pd.DateOffset(years=window_years)
    post_days = int(config.config.get("post_event_training_days", 10))
    selected = events[
        events["stage_date"].ge(start) & events["stage_date"].lt(cutoff)
    ]
    parts: list[pd.DataFrame] = []
    for event in selected.itertuples(index=False):
        subset = daily[
            daily["location_id"].eq(str(event.location_id))
            & daily["date"].dt.year.eq(int(event.stage_date.year))
            & daily["date"].le(event.stage_date + pd.Timedelta(days=post_days))
        ].copy()
        if subset.empty:
            continue
        subset["stage_id"] = int(event.stage_id)
        subset[config.target] = (event.stage_date - subset["date"]).dt.days.astype(float)
        parts.append(subset)
    if not parts:
        raise ValueError(f"No training rows exist before cutoff {cutoff.date()}")
    matrix = pd.concat(parts, ignore_index=True).sort_values("date")
    matrix = matrix.dropna(subset=[*config.features, config.target])
    x_train = matrix.set_index("date")[config.features]
    y_train = matrix.set_index("date")[[config.target]]
    return x_train, y_train


def build_trajectory(
    daily: pd.DataFrame,
    events: pd.DataFrame,
    year: int,
    stage_ids: Iterable[int],
    through_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    end = pd.Timestamp(through_date) if through_date is not None else pd.Timestamp(f"{year}-06-30")
    location_ids = events.loc[events["event_year"].eq(year), "location_id"].unique()
    base = daily[
        daily["date"].dt.year.eq(year)
        & daily["location_id"].isin(location_ids)
        & daily["date"].le(end)
    ].copy()
    frames = [base.assign(stage_id=int(stage_id)) for stage_id in stage_ids]
    return pd.concat(frames, ignore_index=True).sort_values(
        ["date", "location_id", "stage_id"]
    )


def predict_trajectory(model_manager, trajectory: pd.DataFrame, config) -> pd.DataFrame:
    raw = np.asarray(model_manager.predict(trajectory[config.features]), dtype=float)
    out = trajectory[["location_id", "place_id", "stage_id", "latitude", "longitude"]].copy()
    out["prediction_date"] = trajectory["date"].to_numpy()
    out["trajectory_year"] = pd.to_datetime(out["prediction_date"]).dt.year
    out["raw_signed_days"] = raw
    out["predicted_days"] = np.clip(raw, 0.0, None)
    out["predicted_stage_date"] = (
        pd.to_datetime(out["prediction_date"])
        + pd.to_timedelta(out["predicted_days"], unit="D")
    )
    return add_stage_output_names(out)


def _actuals(events: pd.DataFrame, year: int) -> pd.DataFrame:
    return (
        events[events["event_year"].eq(year)]
        .sort_values("stage_date")
        .drop_duplicates(["location_id", "stage_id"])[
            ["location_id", "stage_id", "stage_date"]
        ]
        .rename(columns={"stage_date": "actual_stage_date"})
    )


def chronological_base_trajectories(
    events: pd.DataFrame,
    daily: pd.DataFrame,
    config: WeevillTrakConfig,
) -> pd.DataFrame:
    stage_ids = tuple(config.config["canonical_events"]["model_stage_ids"])
    frames: list[pd.DataFrame] = []
    for year in sorted(events["event_year"].unique()):
        cutoff = pd.Timestamp(f"{year}-01-01")
        if events[events["stage_date"].lt(cutoff)].empty:
            continue
        try:
            x_train, y_train = build_training_matrix(events, daily, config, cutoff)
        except ValueError:
            continue
        manager = create_model_manager(config, S3Manager("local"))
        manager.train(x_train, y_train)
        trajectory = build_trajectory(daily, events, int(year), stage_ids)
        if trajectory.empty:
            continue
        predicted = predict_trajectory(manager, trajectory, config)
        predicted = predicted.merge(
            _actuals(events, int(year)), on=["location_id", "stage_id"], how="inner"
        )
        predicted["actual_days_to_event"] = (
            pd.to_datetime(predicted["actual_stage_date"])
            - pd.to_datetime(predicted["prediction_date"])
        ).dt.total_seconds() / 86400.0
        frames.append(predicted)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def train_local(
    config_path: str | Path = DEFAULT_CONFIG,
    events_path: str | Path = DEFAULT_EVENTS,
    features_path: str | Path = DEFAULT_FEATURES,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
) -> dict:
    config_file = _resolve(config_path)
    event_file = _resolve(events_path)
    feature_file = _resolve(features_path)
    config = WeevillTrakConfig(str(config_file))
    events = add_configured_manual_events(load_events(event_file), config)
    daily = load_engineered_features(feature_file, config.features)
    cutoff = events["stage_date"].max().normalize() + pd.Timedelta(days=1)
    policy = RecentLabelCoveragePolicy.from_config(config.config)
    coverage = require_recent_label_coverage(
        events,
        cutoff_date=cutoff.strftime("%Y-%m-%d"),
        window_years=int(config.config["training_window_years"]),
        stage_ids=tuple(config.config["canonical_events"]["model_stage_ids"]),
        policy=policy,
    )
    x_train, y_train = build_training_matrix(events, daily, config, cutoff)
    manager = create_model_manager(config, S3Manager("local"))
    base_model = manager.train(x_train, y_train)

    oof = chronological_base_trajectories(events, daily, config)
    if oof.empty:
        raise ValueError("No chronological OOF trajectories available for termination training")
    settings = LearnedTerminationSettings.from_config(config.config)
    termination_model = train_learned_termination_model(
        oof,
        oof["actual_days_to_event"].to_numpy(float),
        alpha=settings.alpha,
    )

    destination = _resolve(artifact_dir)
    destination.mkdir(parents=True, exist_ok=True)
    base_path = destination / "base_model.pkl"
    termination_path = destination / "termination_model.pkl"
    with base_path.open("wb") as handle:
        pickle.dump(base_model, handle)
    with termination_path.open("wb") as handle:
        pickle.dump(termination_model, handle)
    frozen_config = destination / "config.yml"
    frozen_config.write_text(config_file.read_text(encoding="utf-8"), encoding="utf-8")
    coverage.to_csv(destination / "label_coverage.csv", index=False)
    manifest = {
        "release": "weeviltrak-v2.4-local",
        "training_cutoff": cutoff.strftime("%Y-%m-%d"),
        "training_rows": int(len(x_train)),
        "oof_termination_rows": int(len(oof)),
        "events_sha256": _sha256(event_file),
        "features_sha256": _sha256(feature_file),
        "artifacts": {
            "base_model.pkl": _sha256(base_path),
            "termination_model.pkl": _sha256(termination_path),
            "config.yml": _sha256(frozen_config),
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def load_local_models(artifact_dir: str | Path):
    source = _resolve(artifact_dir)
    config = WeevillTrakConfig(str(source / "config.yml"))
    manager = create_model_manager(config, S3Manager("local"))
    with (source / "base_model.pkl").open("rb") as handle:
        manager.model = pickle.load(handle)
    with (source / "termination_model.pkl").open("rb") as handle:
        termination = pickle.load(handle)
    return config, manager, termination


def predict_local(
    prediction_date: str,
    features_path: str | Path = DEFAULT_FEATURES,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    events_path: str | Path = DEFAULT_EVENTS,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    date = pd.Timestamp(prediction_date).normalize()
    config, manager, termination = load_local_models(artifact_dir)
    events = add_configured_manual_events(load_events(events_path), config)
    daily = load_engineered_features(features_path, config.features)
    trajectory = build_trajectory(
        daily,
        events,
        date.year,
        config.config["canonical_events"]["model_stage_ids"],
        through_date=date,
    )
    if trajectory.empty or not trajectory["date"].eq(date).any():
        raise ValueError(f"No local feature rows are available for {date.date()}")
    out = predict_trajectory(manager, trajectory, config)
    settings = LearnedTerminationSettings.from_config(config.config)
    out = apply_learned_termination_rule(
        out,
        termination,
        threshold=settings.threshold,
        overwrite_business_prediction=settings.overwrite_business_prediction,
        min_history_days=settings.min_history_days,
    )
    today = out[pd.to_datetime(out["prediction_date"]).eq(date)].copy()
    if output_path is not None:
        destination = _resolve(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        today.to_csv(destination, index=False)
    return today.sort_values(["location_id", "output_event_order"])


def walk_forward_local(
    config_path: str | Path = DEFAULT_CONFIG,
    events_path: str | Path = DEFAULT_EVENTS,
    features_path: str | Path = DEFAULT_FEATURES,
    output_dir: str | Path = DEFAULT_BACKTEST_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = WeevillTrakConfig(str(_resolve(config_path)))
    events = add_configured_manual_events(load_events(events_path), config)
    daily = load_engineered_features(features_path, config.features)
    detail = chronological_base_trajectories(events, daily, config)
    if detail.empty:
        raise ValueError("No walk-forward folds could be built")
    settings = LearnedTerminationSettings.from_config(config.config)
    completed: list[pd.DataFrame] = []
    for year in sorted(detail["trajectory_year"].unique()):
        current = detail[detail["trajectory_year"].eq(year)].copy()
        history = detail[detail["trajectory_year"].lt(year)].copy()
        if not history.empty:
            termination = train_learned_termination_model(
                history,
                history["actual_days_to_event"].to_numpy(float),
                alpha=settings.alpha,
            )
            current = apply_learned_termination_rule(
                current,
                termination,
                threshold=settings.threshold,
                overwrite_business_prediction=settings.overwrite_business_prediction,
                min_history_days=settings.min_history_days,
            )
        current["error_days"] = (
            pd.to_datetime(current["predicted_stage_date"])
            - pd.to_datetime(current["actual_stage_date"])
        ).dt.total_seconds() / 86400.0
        completed.append(current)
    detail = pd.concat(completed, ignore_index=True)
    summary = (
        detail.groupby(["trajectory_year", "stage_id"], as_index=False)
        .agg(
            n=("error_days", "size"),
            mae=("error_days", lambda value: float(np.mean(np.abs(value)))),
            rmse=("error_days", lambda value: float(np.sqrt(np.mean(value**2)))),
            bias=("error_days", "mean"),
        )
        .sort_values(["trajectory_year", "stage_id"])
    )
    destination = _resolve(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    detail.to_csv(destination / "walk_forward_detail.csv", index=False)
    summary.to_csv(destination / "walk_forward_summary.csv", index=False)
    return detail, summary

"""Generate standard unified historical trajectory caches.

This script creates reusable v2.4 historical/cache artifacts under
``outputs/backtesting/v2.4``:

    - actual_events.parquet / .csv
    - trajectory_detail.parquet / .csv
    - termination_event_detail.parquet / .csv
    - summary_year_stage.csv
    - run_metadata.json

The cache is intended to be a shared input for notebook experiments such as
Stage-to-Phase / Method C calibration.  For a learned-termination config, the
termination model for each validation year is fit only on earlier out-of-fold
trajectory years.  It must never load the final production Ridge artifact,
because that artifact can contain information from the validation year.

Example:

    poetry run python scripts/backtesting/generate_v24_unified_historical_cache.py

To force a clean rebuild:

    poetry run python scripts/backtesting/generate_v24_unified_historical_cache.py --force-rerun
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline.backtesting import BacktestingFramework
from app.settings import setup_logger

logger = setup_logger()


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "app/config/weeviltrak_v2.4.yml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/backtesting/v2.4"
DEFAULT_YEARS = [2023, 2024, 2025]
DEFAULT_STAGES = [1, 4, 2, 3]


@dataclass(frozen=True)
class V24HistoricalCacheConfig:
    years: list[int]
    stages: list[int]
    prediction_interval_days: int = 3
    season_start_md: str = "03-01"
    season_end_md: str = "06-01"
    include_actual_stage_dates: bool = True
    training_window_years: int | None = None
    force_expanding_window: bool = False
    max_locations_per_year: int | None = None
    random_seed: int = 42


def _parse_int_list(raw_values: Iterable[str] | None, default: list[int]) -> list[int]:
    if not raw_values:
        return default
    parsed: list[int] = []
    for raw in raw_values:
        for part in raw.split(","):
            part = part.strip()
            if part:
                parsed.append(int(part))
    return list(dict.fromkeys(parsed)) or default


def normalize_actual_events(
    weevil_data: pd.DataFrame,
    years: list[int],
    stages: list[int],
) -> pd.DataFrame:
    events = weevil_data.copy()
    events["stage_date"] = pd.to_datetime(events["stage_date"], errors="coerce").dt.normalize()
    events["location_id"] = events["location_id"].astype(str)
    events["stage_id"] = pd.to_numeric(events["stage_id"], errors="coerce").astype(int)
    events["year"] = pd.to_numeric(events["year"], errors="coerce").astype(int)

    keep_cols = [
        "location_id",
        "stage_id",
        "year",
        "stage_date",
        "latitude",
        "longitude",
    ]
    keep_cols += [
        col
        for col in ["location_name", "location_state", "state_code"]
        if col in events.columns
    ]
    events = events[events["year"].isin(years) & events["stage_id"].isin(stages)][
        keep_cols
    ].copy()
    events = (
        events.sort_values(["year", "location_id", "stage_id", "stage_date"])
        .drop_duplicates(["year", "location_id", "stage_id"], keep="first")
        .reset_index(drop=True)
    )
    return events


def build_prediction_dates_for_year(
    year: int,
    actual_events_for_year: pd.DataFrame,
    config: V24HistoricalCacheConfig,
) -> list[pd.Timestamp]:
    start = pd.Timestamp(f"{year}-{config.season_start_md}")
    end = pd.Timestamp(f"{year}-{config.season_end_md}")
    dates = set(
        pd.date_range(
            start,
            end,
            freq=f"{config.prediction_interval_days}D",
        ).normalize()
    )

    if config.include_actual_stage_dates and not actual_events_for_year.empty:
        event_dates = pd.to_datetime(
            actual_events_for_year["stage_date"], errors="coerce"
        ).dt.normalize()
        event_dates = event_dates[(event_dates >= start) & (event_dates <= end)]
        dates.update(event_dates.dropna().tolist())

    return sorted(dates)


def select_locations_for_year(
    events_for_year: pd.DataFrame,
    year: int,
    config: V24HistoricalCacheConfig,
) -> list[str]:
    locations = sorted(events_for_year["location_id"].astype(str).unique().tolist())
    if (
        config.max_locations_per_year is None
        or len(locations) <= config.max_locations_per_year
    ):
        return locations
    rng = np.random.default_rng(config.random_seed + int(year))
    return sorted(
        rng.choice(locations, size=config.max_locations_per_year, replace=False).tolist()
    )


def build_schedule(
    actual_events: pd.DataFrame,
    config: V24HistoricalCacheConfig,
) -> tuple[dict[int, list[pd.Timestamp]], dict[int, list[str]], pd.DataFrame]:
    prediction_dates_by_year: dict[int, list[pd.Timestamp]] = {}
    locations_by_year: dict[int, list[str]] = {}
    rows: list[dict[str, object]] = []

    for year in config.years:
        events_for_year = actual_events[actual_events["year"] == year].copy()
        prediction_dates = build_prediction_dates_for_year(year, events_for_year, config)
        locations = select_locations_for_year(events_for_year, year, config)
        prediction_dates_by_year[year] = prediction_dates
        locations_by_year[year] = locations
        rows.append(
            {
                "test_year": int(year),
                "training_cutoff": f"{year}-01-01",
                "train_data_through": f"{year - 1}-12-31",
                "n_prediction_dates": len(prediction_dates),
                "first_prediction_date": min(prediction_dates) if prediction_dates else pd.NaT,
                "last_prediction_date": max(prediction_dates) if prediction_dates else pd.NaT,
                "n_locations": len(locations),
            }
        )
    return prediction_dates_by_year, locations_by_year, pd.DataFrame(rows)


def add_trajectory_diagnostics(detail: pd.DataFrame) -> pd.DataFrame:
    df = detail.copy()
    df["prediction_date"] = pd.to_datetime(df["prediction_date"], errors="coerce").dt.normalize()
    df["actual_stage_date"] = pd.to_datetime(df["actual_stage_date"], errors="coerce").dt.normalize()
    df["predicted_stage_date"] = pd.to_datetime(
        df["predicted_stage_date"], errors="coerce"
    ).dt.normalize()
    df["location_id"] = df["location_id"].astype(str)
    df["stage_id"] = pd.to_numeric(df["stage_id"], errors="coerce").astype(int)
    df["test_year"] = pd.to_numeric(df["test_year"], errors="coerce").astype(int)
    df["predicted_days"] = pd.to_numeric(df["predicted_days"], errors="coerce")

    df["days_from_event"] = (df["prediction_date"] - df["actual_stage_date"]).dt.days
    df["actual_days_to_event"] = (df["actual_stage_date"] - df["prediction_date"]).dt.days
    df["stage_date_error_days"] = (
        df["predicted_stage_date"] - df["actual_stage_date"]
    ).dt.days
    df["abs_stage_date_error_days"] = df["stage_date_error_days"].abs()
    df["within_3d"] = df["abs_stage_date_error_days"] <= 3
    df["within_5d"] = df["abs_stage_date_error_days"] <= 5
    df["within_7d"] = df["abs_stage_date_error_days"] <= 7

    group_cols = ["model_version", "test_year", "location_id", "stage_id"]
    df = df.sort_values(group_cols + ["prediction_date"]).reset_index(drop=True)
    df["predicted_days_delta"] = df.groupby(group_cols)["predicted_days"].diff()
    df["predicted_stage_date_delta_days"] = (
        df.groupby(group_cols)["predicted_stage_date"].diff().dt.days
    )
    df["abs_predicted_stage_date_drift_days"] = df[
        "predicted_stage_date_delta_days"
    ].abs()
    df["bounce_up"] = df["predicted_days_delta"] > 0
    df["post_event_positive"] = (df["days_from_event"] > 0) & (df["predicted_days"] > 0)
    return df


def build_termination_event_detail(trajectory_detail: pd.DataFrame) -> pd.DataFrame:
    if trajectory_detail.empty:
        return pd.DataFrame()

    df = trajectory_detail.copy()
    for col in [
        "prediction_date",
        "actual_stage_date",
        "predicted_stage_date",
        "termination_decision_date",
        "termination_estimated_stage_date",
    ]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
    df["location_id"] = df["location_id"].astype(str)
    df["stage_id"] = pd.to_numeric(df["stage_id"], errors="coerce").astype(int)
    df["test_year"] = pd.to_numeric(df["test_year"], errors="coerce").astype(int)

    rows: list[dict[str, object]] = []
    group_cols = ["model_version", "test_year", "location_id", "stage_id"]
    optional_cols = [
        "location_name",
        "location_state",
        "actual_latitude",
        "actual_longitude",
        "latitude",
        "longitude",
    ]
    present_optional = [c for c in optional_cols if c in df.columns]

    for key, group in df.groupby(group_cols, sort=True, dropna=False):
        model_version, test_year, location_id, stage_id = key
        group = group.sort_values("prediction_date").copy()
        actual_date = group["actual_stage_date"].dropna().iloc[0]
        reached_series = (
            group["termination_reached"].fillna(False).astype(bool)
            if "termination_reached" in group.columns
            else pd.Series(False, index=group.index)
        )
        reached = group[reached_series]
        if not reached.empty:
            chosen = reached.iloc[0]
            decision_date = chosen.get("termination_decision_date", chosen["prediction_date"])
            estimated_date = chosen.get(
                "termination_estimated_stage_date", chosen["predicted_stage_date"]
            )
            reached_flag = True
        else:
            chosen = group.iloc[-1]
            decision_date = pd.NaT
            estimated_date = chosen.get("predicted_stage_date", pd.NaT)
            reached_flag = False

        error_days = (
            (pd.to_datetime(decision_date) - actual_date).days
            if pd.notna(decision_date)
            else np.nan
        )
        row: dict[str, object] = {
            "model_version": model_version,
            "test_year": int(test_year),
            "location_id": str(location_id),
            "stage_id": int(stage_id),
            "actual_stage_date": actual_date,
            "termination_reached": bool(reached_flag),
            "termination_decision_date": decision_date,
            "termination_estimated_stage_date": estimated_date,
            "termination_error_days": error_days,
            "abs_termination_error_days": abs(error_days) if pd.notna(error_days) else np.nan,
            "termination_early": bool(pd.notna(error_days) and error_days < 0),
            "termination_late": bool(pd.notna(error_days) and error_days > 0),
            "selection_rule": (
                "first_termination_reached"
                if reached_flag
                else "last_available_prediction_no_termination"
            ),
        }
        for col in present_optional:
            values = group[col].dropna()
            row[col] = values.iloc[0] if not values.empty else np.nan
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["test_year", "stage_id", "location_id"])


def apply_legacy_zero_day_termination(
    trajectory_detail: pd.DataFrame,
    threshold: float,
    consecutive_days: int,
) -> pd.DataFrame:
    """Apply the pre-v2.4 zero-day business rule to legacy trajectories.

    v2.3 does not have a learned termination model, but its production
    behavior locked a trajectory at zero after a consecutive near-zero run.
    Populate the standard termination fields so cache consumers can compare
    legacy and learned-termination versions without treating missing columns
    as an observed outcome.
    """
    if trajectory_detail.empty or "termination_reached" in trajectory_detail:
        return trajectory_detail

    df = trajectory_detail.copy()
    df["prediction_date"] = pd.to_datetime(df["prediction_date"], errors="coerce")
    df["predicted_stage_date"] = pd.to_datetime(
        df["predicted_stage_date"], errors="coerce"
    )
    df["termination_reached"] = False
    df["termination_decision_date"] = pd.NaT
    df["termination_estimated_stage_date"] = pd.NaT

    group_cols = ["model_version", "test_year", "location_id", "stage_id"]
    for _, group in df.groupby(group_cols, sort=False, dropna=False):
        ordered = group.sort_values("prediction_date")
        near_zero = pd.to_numeric(ordered["predicted_days"], errors="coerce") <= threshold
        run = 0
        lock_index = None
        run_start_index = None
        for index, is_near_zero in near_zero.items():
            if bool(is_near_zero):
                if run == 0:
                    run_start_index = index
                run += 1
            else:
                run = 0
                run_start_index = None
            if run >= consecutive_days:
                lock_index = run_start_index
                break
        if lock_index is None:
            continue

        decision_date = df.at[lock_index, "prediction_date"]
        locked_rows = ordered.loc[ordered["prediction_date"] >= decision_date].index
        df.loc[locked_rows, "predicted_days"] = 0.0
        df.loc[locked_rows, "predicted_stage_date"] = decision_date
        df.loc[locked_rows, "termination_reached"] = True
        df.loc[locked_rows, "termination_decision_date"] = decision_date
        df.loc[locked_rows, "termination_estimated_stage_date"] = decision_date

    return df


def summarize_year_stage(trajectory_detail: pd.DataFrame) -> pd.DataFrame:
    if trajectory_detail.empty:
        return pd.DataFrame()

    grouped = trajectory_detail.groupby(["model_version", "test_year", "stage_id"])
    return (
        grouped.agg(
            n_prediction_rows=("location_id", "size"),
            n_trajectories=("location_id", "nunique"),
            bias_stage_date_error=("stage_date_error_days", "mean"),
            mae_stage_date_error=("abs_stage_date_error_days", "mean"),
            rmse_stage_date_error=(
                "stage_date_error_days",
                lambda s: float(np.sqrt(np.mean(pd.to_numeric(s, errors="coerce").dropna() ** 2))),
            ),
            median_abs_stage_date_error=("abs_stage_date_error_days", "median"),
            within_3d_rate=("within_3d", "mean"),
            within_5d_rate=("within_5d", "mean"),
            within_7d_rate=("within_7d", "mean"),
            bounce_up_rate=("bounce_up", "mean"),
            post_event_positive_rate=("post_event_positive", "mean"),
            termination_reached_rate=("termination_reached", "mean"),
        )
        .reset_index()
        .sort_values(["model_version", "test_year", "stage_id"])
    )


def save_frame(df: pd.DataFrame, base_path: Path) -> None:
    df.to_parquet(base_path.with_suffix(".parquet"), index=False)
    df.to_csv(base_path.with_suffix(".csv"), index=False)


def apply_evaluation_termination(
    backtester: BacktestingFramework,
    trajectory_detail: pd.DataFrame,
    weevil_data: pd.DataFrame,
) -> pd.DataFrame:
    """Apply termination without leaking a final production artifact.

    ``BacktestingFramework._apply_oof_termination_to_backtest_detail`` keeps
    each test year isolated from its own and later actual events.  This is the
    only valid learned-termination path for a historical cache.
    """
    return backtester._apply_oof_termination_to_backtest_detail(
        trajectory_detail,
        weevil_data,
    )


def generate_v24_unified_historical_cache(
    config_path: Path,
    output_dir: Path,
    cache_config: V24HistoricalCacheConfig,
    force_rerun: bool = False,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    actual_events_path = output_dir / "actual_events.parquet"
    trajectory_path = output_dir / "trajectory_detail.parquet"
    termination_path = output_dir / "termination_event_detail.parquet"
    summary_path = output_dir / "summary_year_stage.csv"
    metadata_path = output_dir / "run_metadata.json"
    weather_cache_path = output_dir / "processed_weather_unique.parquet"
    schedule_path = output_dir / "prediction_schedule.csv"

    if (
        trajectory_path.exists()
        and termination_path.exists()
        and actual_events_path.exists()
        and not force_rerun
    ):
        logger.info("Unified v2.4 cache already exists under %s", output_dir)
        logger.info("Use --force-rerun to rebuild.")
        return {
            "actual_events": actual_events_path,
            "trajectory_detail": trajectory_path,
            "termination_event_detail": termination_path,
            "summary_year_stage": summary_path,
            "metadata": metadata_path,
        }

    backtester = BacktestingFramework(
        config_path=str(config_path),
        output_dir=str(output_dir),
    )
    model_version = str(backtester.config.config.get("version") or "unknown")

    latest_pull_date = f"{max(cache_config.years)}-12-31"
    weevil_data = backtester.data_service.pull_weevil_data(today=latest_pull_date)
    actual_events = normalize_actual_events(
        weevil_data=weevil_data,
        years=cache_config.years,
        stages=cache_config.stages,
    )
    prediction_dates_by_year, locations_by_year, schedule = build_schedule(
        actual_events=actual_events,
        config=cache_config,
    )

    actual_events.to_parquet(actual_events_path, index=False)
    actual_events.to_csv(output_dir / "actual_events.csv", index=False)
    schedule.to_csv(schedule_path, index=False)

    all_prediction_dates = sorted(
        {date for dates in prediction_dates_by_year.values() for date in dates}
    )
    if weather_cache_path.exists() and not force_rerun:
        logger.info("Loading cached processed weather: %s", weather_cache_path)
        processed_weather_unique = pd.read_parquet(weather_cache_path)
    else:
        processed_weather_unique = backtester.prepare_weather(
            weevil_data=weevil_data,
            test_dates=all_prediction_dates,
        )
        processed_weather_unique.to_parquet(weather_cache_path, index=False)

    actual_optional_cols = [
        col
        for col in ["location_name", "location_state", "state_code"]
        if col in actual_events.columns
    ]
    actual_key = actual_events[
        [
            "year",
            "location_id",
            "stage_id",
            "stage_date",
            "latitude",
            "longitude",
        ]
        + actual_optional_cols
    ].copy()
    actual_key = actual_key.rename(
        columns={
            "year": "test_year",
            "stage_date": "actual_stage_date",
            "latitude": "actual_latitude",
            "longitude": "actual_longitude",
        }
    )

    detail_parts: list[pd.DataFrame] = []
    for year in cache_config.years:
        cutoff_date = pd.Timestamp(f"{year}-01-01")
        cutoff_str = cutoff_date.strftime("%Y-%m-%d")
        prediction_dates = prediction_dates_by_year[year]
        locations = locations_by_year[year]
        logger.info(
            "[%s] Year %s: cutoff=%s dates=%s locations=%s training_window_years=%s",
            model_version,
            year,
            cutoff_str,
            len(prediction_dates),
            len(locations),
            cache_config.training_window_years or "config_default",
        )
        model = backtester.train_model_for_cutoff(
            cutoff_date=cutoff_date,
            processed_weather_unique=processed_weather_unique,
            training_window_years=cache_config.training_window_years,
            use_config_default=not cache_config.force_expanding_window,
        )
        year_parts: list[pd.DataFrame] = []
        for i, prediction_date in enumerate(prediction_dates, start=1):
            out = backtester.predict_with_model_for_date(
                model=model,
                test_date=prediction_date,
                locations_for_date=locations,
                model_train_cutoff_date=cutoff_str,
                processed_weather_unique=processed_weather_unique,
            )
            if out is None or out.empty:
                continue
            if out.duplicated(
                subset=["prediction_date", "location_id", "stage_id"], keep=False
            ).any():
                out = backtester.aggregate_predictions_unique(out)
            out = out.copy()
            out["model_version"] = model_version
            out["test_year"] = int(year)
            year_parts.append(out)
            if i % 10 == 0 or i == len(prediction_dates):
                logger.info(
                    "[%s] Year %s: %s/%s dates complete",
                    model_version,
                    year,
                    i,
                    len(prediction_dates),
                )
        if not year_parts:
            logger.warning("[%s] Year %s produced no predictions", model_version, year)
            continue
        year_predictions = pd.concat(year_parts, ignore_index=True)
        year_predictions["location_id"] = year_predictions["location_id"].astype(str)
        year_predictions["stage_id"] = pd.to_numeric(
            year_predictions["stage_id"], errors="coerce"
        ).astype(int)
        year_predictions["prediction_date"] = pd.to_datetime(
            year_predictions["prediction_date"], errors="coerce"
        ).dt.normalize()
        year_detail = year_predictions.merge(
            actual_key,
            on=["test_year", "location_id", "stage_id"],
            how="inner",
        )
        detail_parts.append(year_detail)

    if not detail_parts:
        raise RuntimeError(
            f"No {model_version} historical trajectory predictions were produced."
        )

    trajectory_detail = pd.concat(detail_parts, ignore_index=True)
    trajectory_detail = apply_evaluation_termination(
        backtester=backtester,
        trajectory_detail=trajectory_detail,
        weevil_data=weevil_data,
    )
    if "termination_reached" not in trajectory_detail:
        trajectory_detail = apply_legacy_zero_day_termination(
            trajectory_detail,
            threshold=float(
                backtester.config.config.get("zero_day_termination_threshold", 1.0)
            ),
            consecutive_days=int(
                backtester.config.config.get("zero_day_termination_consecutive_days", 2)
            ),
        )
    trajectory_detail = add_trajectory_diagnostics(trajectory_detail)
    termination_event_detail = build_termination_event_detail(trajectory_detail)
    summary_year_stage = summarize_year_stage(trajectory_detail)

    save_frame(trajectory_detail, output_dir / "trajectory_detail")
    save_frame(termination_event_detail, output_dir / "termination_event_detail")
    summary_year_stage.to_csv(summary_path, index=False)

    metadata = {
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "config_path": str(config_path.relative_to(PROJECT_ROOT) if config_path.is_relative_to(PROJECT_ROOT) else config_path),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT) if output_dir.is_relative_to(PROJECT_ROOT) else output_dir),
        "cache_config": asdict(cache_config),
        "resolved_training_window_years": (
            None
            if cache_config.force_expanding_window
            else backtester.resolve_training_window_years(
                cache_config.training_window_years
            )
        ),
        "config_version": model_version,
        "termination_model": backtester.config.config.get("termination_model", {}),
        "termination_evaluation_mode": "walk_forward_oof",
        "n_actual_events": int(len(actual_events)),
        "n_trajectory_rows": int(len(trajectory_detail)),
        "n_termination_events": int(len(termination_event_detail)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

    return {
        "actual_events": actual_events_path,
        "trajectory_detail": trajectory_path,
        "termination_event_detail": termination_path,
        "summary_year_stage": summary_path,
        "metadata": metadata_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate standard unified v2.4 historical trajectory caches."
    )
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--year", dest="years", action="append", help="Year(s), repeat or comma-separated.")
    parser.add_argument("--stage", dest="stages", action="append", help="Stage(s), repeat or comma-separated.")
    parser.add_argument("--prediction-interval-days", type=int, default=3)
    parser.add_argument("--season-start-md", default="03-01")
    parser.add_argument("--season-end-md", default="06-01")
    parser.add_argument("--exclude-actual-stage-dates", action="store_true")
    window_group = parser.add_mutually_exclusive_group()
    window_group.add_argument(
        "--training-window-years",
        type=int,
        default=None,
        help="Override config default. Omit to use v2.4 config default, currently 3.",
    )
    window_group.add_argument(
        "--expanding-window",
        action="store_true",
        help="Explicitly use all data before each cutoff, ignoring a config default.",
    )
    parser.add_argument("--max-locations-per-year", type=int)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--force-rerun", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cache_config = V24HistoricalCacheConfig(
        years=_parse_int_list(args.years, DEFAULT_YEARS),
        stages=_parse_int_list(args.stages, DEFAULT_STAGES),
        prediction_interval_days=args.prediction_interval_days,
        season_start_md=args.season_start_md,
        season_end_md=args.season_end_md,
        include_actual_stage_dates=not args.exclude_actual_stage_dates,
        training_window_years=args.training_window_years,
        force_expanding_window=args.expanding_window,
        max_locations_per_year=args.max_locations_per_year,
        random_seed=args.random_seed,
    )
    outputs = generate_v24_unified_historical_cache(
        config_path=args.config_path,
        output_dir=args.output_dir,
        cache_config=cache_config,
        force_rerun=args.force_rerun,
    )
    print("DONE")
    for name, path in outputs.items():
        print(f"{name.upper()}={path}")


if __name__ == "__main__":
    main()

"""Run v2.2/v2.3 trajectory backtesting and compute zero-day error.

Zero-day error is defined per model/year/location/stage trajectory as:

    first prediction_date where predicted_days <= zero_threshold
    minus actual_stage_date

Negative values mean the model first declared zero days too early. Positive
values mean it declared zero days too late.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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


DEFAULT_YEARS = [2022, 2023, 2024, 2025]
DEFAULT_STAGES = [1, 2, 3]
DEFAULT_MODEL_CONFIGS = {
    "v2.2": PROJECT_ROOT / "app/config/weeviltrak_v2.2.yml",
    "v2.3": PROJECT_ROOT / "app/config/weeviltrak_v2.3.yml",
}


@dataclass(frozen=True)
class ZeroDayBacktestConfig:
    years: list[int]
    stages: list[int]
    prediction_interval_days: int = 3
    season_start_md: str = "03-01"
    season_end_md: str = "06-01"
    include_actual_stage_dates: bool = True
    max_locations_per_year: int | None = None
    random_seed: int = 42
    zero_threshold: float = 0.0
    training_window_years: int | None = None


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


def build_prediction_dates_for_year(
    year: int,
    actual_events_for_year: pd.DataFrame,
    config: ZeroDayBacktestConfig,
) -> list[pd.Timestamp]:
    """Build the same 3-day seasonal grid used by trajectory_backtest_v22_v23."""
    start = pd.Timestamp(f"{year}-{config.season_start_md}")
    end = pd.Timestamp(f"{year}-{config.season_end_md}")
    grid = set(
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
        grid.update(event_dates.dropna().tolist())

    return sorted(grid)


def normalize_actual_events(
    weevil_data: pd.DataFrame,
    years: list[int],
    stages: list[int],
) -> pd.DataFrame:
    """Normalize and deduplicate actual stage events for trajectory evaluation."""
    events = weevil_data.copy()
    events["stage_date"] = pd.to_datetime(events["stage_date"], errors="coerce").dt.normalize()
    events["location_id"] = events["location_id"].astype(str)
    events["stage_id"] = pd.to_numeric(events["stage_id"], errors="coerce").astype(int)
    events["year"] = pd.to_numeric(events["year"], errors="coerce").astype(int)

    keep_cols = ["location_id", "stage_id", "year", "stage_date", "latitude", "longitude"]
    optional_cols = ["location_name", "location_state", "state_code"]
    keep_cols += [col for col in optional_cols if col in events.columns]

    events = events[
        events["year"].isin(years) & events["stage_id"].isin(stages)
    ][keep_cols].copy()
    events = (
        events.sort_values(["year", "location_id", "stage_id", "stage_date"])
        .drop_duplicates(["year", "location_id", "stage_id"], keep="first")
        .reset_index(drop=True)
    )
    return events


def select_locations_for_year(
    events_for_year: pd.DataFrame,
    year: int,
    config: ZeroDayBacktestConfig,
) -> list[str]:
    """Select the same location universe as the trajectory notebook."""
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
    config: ZeroDayBacktestConfig,
) -> tuple[dict[int, list[pd.Timestamp]], dict[int, list[str]], pd.DataFrame]:
    prediction_dates_by_year: dict[int, list[pd.Timestamp]] = {}
    locations_by_year: dict[int, list[str]] = {}
    schedule_rows: list[dict[str, object]] = []

    for year in config.years:
        events_for_year = actual_events[actual_events["year"] == year].copy()
        prediction_dates = build_prediction_dates_for_year(year, events_for_year, config)
        locations = select_locations_for_year(events_for_year, year, config)
        prediction_dates_by_year[year] = prediction_dates
        locations_by_year[year] = locations
        schedule_rows.append(
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

    return prediction_dates_by_year, locations_by_year, pd.DataFrame(schedule_rows)


def compute_zero_day_error_detail(
    trajectory_detail: pd.DataFrame,
    zero_threshold: float = 0.0,
) -> pd.DataFrame:
    """Compute first-zero prediction date and zero-day error per trajectory."""
    if trajectory_detail.empty:
        return pd.DataFrame()

    df = trajectory_detail.copy()
    df["prediction_date"] = pd.to_datetime(df["prediction_date"], errors="coerce").dt.normalize()
    df["actual_stage_date"] = pd.to_datetime(
        df["actual_stage_date"], errors="coerce"
    ).dt.normalize()
    df["predicted_stage_date"] = pd.to_datetime(
        df["predicted_stage_date"], errors="coerce"
    ).dt.normalize()
    df["location_id"] = df["location_id"].astype(str)
    df["stage_id"] = pd.to_numeric(df["stage_id"], errors="coerce").astype(int)
    df["test_year"] = pd.to_numeric(df["test_year"], errors="coerce").astype(int)
    df["predicted_days"] = pd.to_numeric(df["predicted_days"], errors="coerce")

    group_cols = ["model_version", "test_year", "location_id", "stage_id"]
    df = df.sort_values(group_cols + ["prediction_date"]).reset_index(drop=True)
    df["is_zero_day_prediction"] = df["predicted_days"] <= float(zero_threshold)

    rows: list[dict[str, object]] = []
    optional_cols = [
        "actual_latitude",
        "actual_longitude",
        "latitude",
        "longitude",
        "location_name",
        "location_state",
        "state_code",
    ]
    present_optional_cols = [col for col in optional_cols if col in df.columns]

    for key, g in df.groupby(group_cols, sort=True, dropna=False):
        model_version, test_year, location_id, stage_id = key
        actual_stage_date = g["actual_stage_date"].dropna().iloc[0]
        zero_rows = g[g["is_zero_day_prediction"]].sort_values("prediction_date")
        first_prediction_date = g["prediction_date"].min()
        last_prediction_date = g["prediction_date"].max()

        row: dict[str, object] = {
            "model_version": model_version,
            "test_year": int(test_year),
            "location_id": str(location_id),
            "stage_id": int(stage_id),
            "actual_stage_date": actual_stage_date,
            "first_prediction_date": first_prediction_date,
            "last_prediction_date": last_prediction_date,
            "n_prediction_rows": int(len(g)),
            "zero_threshold": float(zero_threshold),
            "zero_day_reached": bool(not zero_rows.empty),
            "first_zero_prediction_date": pd.NaT,
            "first_zero_predicted_days": np.nan,
            "first_zero_predicted_stage_date": pd.NaT,
            "zero_day_error_days": np.nan,
            "abs_zero_day_error_days": np.nan,
        }

        if not zero_rows.empty:
            first_zero = zero_rows.iloc[0]
            zero_day_error = (
                first_zero["prediction_date"] - actual_stage_date
            ).days
            row.update(
                {
                    "first_zero_prediction_date": first_zero["prediction_date"],
                    "first_zero_predicted_days": float(first_zero["predicted_days"]),
                    "first_zero_predicted_stage_date": first_zero["predicted_stage_date"],
                    "zero_day_error_days": int(zero_day_error),
                    "abs_zero_day_error_days": abs(int(zero_day_error)),
                }
            )

        for col in present_optional_cols:
            value = g[col].dropna()
            row[col] = value.iloc[0] if not value.empty else np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    out["zero_day_error_direction"] = np.select(
        [
            out["zero_day_error_days"].lt(0),
            out["zero_day_error_days"].gt(0),
            out["zero_day_error_days"].eq(0),
        ],
        ["early", "late", "exact"],
        default="not_reached",
    )
    return out.sort_values(["model_version", "test_year", "stage_id", "location_id"])


def summarize_zero_day_errors(zero_day_detail: pd.DataFrame) -> pd.DataFrame:
    """Summarize zero-day error by model version, year, and stage."""
    if zero_day_detail.empty:
        return pd.DataFrame()

    def _p90_abs(s: pd.Series) -> float:
        values = pd.to_numeric(s, errors="coerce").dropna()
        return float(np.percentile(values, 90)) if len(values) else np.nan

    grouped = zero_day_detail.groupby(["model_version", "test_year", "stage_id"], dropna=False)
    summary = grouped.agg(
        n_trajectories=("location_id", "size"),
        zero_day_reached_count=("zero_day_reached", "sum"),
        zero_day_reached_rate=("zero_day_reached", "mean"),
        mean_zero_day_error_days=("zero_day_error_days", "mean"),
        median_zero_day_error_days=("zero_day_error_days", "median"),
        mean_abs_zero_day_error_days=("abs_zero_day_error_days", "mean"),
        median_abs_zero_day_error_days=("abs_zero_day_error_days", "median"),
        p90_abs_zero_day_error_days=("abs_zero_day_error_days", _p90_abs),
        early_zero_count=("zero_day_error_days", lambda s: int((s < 0).sum())),
        exact_zero_count=("zero_day_error_days", lambda s: int((s == 0).sum())),
        late_zero_count=("zero_day_error_days", lambda s: int((s > 0).sum())),
    ).reset_index()

    reached = summary["zero_day_reached_count"].replace(0, np.nan)
    summary["early_zero_rate_among_reached"] = summary["early_zero_count"] / reached
    summary["exact_zero_rate_among_reached"] = summary["exact_zero_count"] / reached
    summary["late_zero_rate_among_reached"] = summary["late_zero_count"] / reached

    return summary.sort_values(["model_version", "test_year", "stage_id"])


def compare_zero_day_versions(summary: pd.DataFrame) -> pd.DataFrame:
    """Create v2.3 minus v2.2 comparison rows for zero-day metrics."""
    if summary.empty:
        return pd.DataFrame()

    metrics = [
        "zero_day_reached_rate",
        "mean_zero_day_error_days",
        "median_zero_day_error_days",
        "mean_abs_zero_day_error_days",
        "median_abs_zero_day_error_days",
        "p90_abs_zero_day_error_days",
        "early_zero_rate_among_reached",
        "exact_zero_rate_among_reached",
        "late_zero_rate_among_reached",
    ]
    rows: list[dict[str, object]] = []
    for (year, stage), g in summary.groupby(["test_year", "stage_id"]):
        by_version = g.set_index("model_version")
        if not {"v2.2", "v2.3"} <= set(by_version.index):
            continue
        row: dict[str, object] = {"test_year": int(year), "stage_id": int(stage)}
        for metric in metrics:
            row[f"{metric}_v2.2"] = by_version.loc["v2.2", metric]
            row[f"{metric}_v2.3"] = by_version.loc["v2.3", metric]
            row[f"{metric}_delta_v23_minus_v22"] = (
                row[f"{metric}_v2.3"] - row[f"{metric}_v2.2"]
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["test_year", "stage_id"]).reset_index(drop=True)


def run_model_zero_day_backtest(
    model_version: str,
    config_path: Path,
    output_root: Path,
    weevil_data: pd.DataFrame,
    actual_events: pd.DataFrame,
    prediction_dates_by_year: dict[int, list[pd.Timestamp]],
    locations_by_year: dict[int, list[str]],
    zero_config: ZeroDayBacktestConfig,
    force_rerun: bool = False,
) -> pd.DataFrame:
    """Run one model version and return per-trajectory zero-day detail."""
    version_dir = output_root / model_version
    version_dir.mkdir(parents=True, exist_ok=True)
    zero_detail_path = version_dir / f"zero_day_error_detail_{model_version}.parquet"
    weather_cache_path = version_dir / "processed_weather_unique.parquet"

    if zero_detail_path.exists() and not force_rerun:
        logger.info("[%s] Loading cached zero-day detail: %s", model_version, zero_detail_path)
        return pd.read_parquet(zero_detail_path)

    backtester = BacktestingFramework(
        config_path=str(config_path),
        output_dir=str(version_dir),
    )
    all_prediction_dates = sorted(
        {date for dates in prediction_dates_by_year.values() for date in dates}
    )

    if weather_cache_path.exists() and not force_rerun:
        logger.info("[%s] Loading cached processed weather: %s", model_version, weather_cache_path)
        processed_weather_unique = pd.read_parquet(weather_cache_path)
    else:
        logger.info("[%s] Preparing weather cache", model_version)
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

    yearly_zero_parts: list[pd.DataFrame] = []
    for year in zero_config.years:
        cutoff_date = pd.Timestamp(f"{year}-01-01")
        cutoff_str = cutoff_date.strftime("%Y-%m-%d")
        locations_for_year = locations_by_year[year]
        prediction_dates = prediction_dates_by_year[year]

        logger.info(
            "[%s] Year %s: train < %s; dates=%s locations=%s",
            model_version,
            year,
            cutoff_str,
            len(prediction_dates),
            len(locations_for_year),
        )
        model = backtester.train_model_for_cutoff(
            cutoff_date=cutoff_date,
            processed_weather_unique=processed_weather_unique,
            training_window_years=zero_config.training_window_years,
        )

        prediction_parts: list[pd.DataFrame] = []
        for i, prediction_date in enumerate(prediction_dates, start=1):
            out = backtester.predict_with_model_for_date(
                model=model,
                test_date=prediction_date,
                locations_for_date=locations_for_year,
                model_train_cutoff_date=cutoff_str,
                processed_weather_unique=processed_weather_unique,
            )
            if out is None or out.empty:
                continue

            key_cols = ["prediction_date", "location_id", "stage_id"]
            if out.duplicated(subset=key_cols, keep=False).any():
                out = backtester.aggregate_predictions_unique(out)

            out = out.copy()
            out["model_version"] = model_version
            out["test_year"] = int(year)
            keep_cols = [
                "model_version",
                "test_year",
                "prediction_date",
                "location_id",
                "stage_id",
                "predicted_days",
                "predicted_stage_date",
                "raw_signed_days",
                "latitude",
                "longitude",
            ]
            prediction_parts.append(out[[col for col in keep_cols if col in out.columns]])

            if i % 10 == 0 or i == len(prediction_dates):
                logger.info(
                    "[%s] Year %s: %s/%s dates complete",
                    model_version,
                    year,
                    i,
                    len(prediction_dates),
                )

        if not prediction_parts:
            logger.warning("[%s] Year %s produced no predictions", model_version, year)
            continue

        year_predictions = pd.concat(prediction_parts, ignore_index=True)
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
        yearly_zero_parts.append(
            compute_zero_day_error_detail(
                year_detail,
                zero_threshold=zero_config.zero_threshold,
            )
        )

    if not yearly_zero_parts:
        raise RuntimeError(f"No zero-day detail produced for {model_version}")

    zero_detail = pd.concat(yearly_zero_parts, ignore_index=True)
    zero_detail.to_parquet(zero_detail_path, index=False)
    zero_detail.to_csv(
        version_dir / f"zero_day_error_detail_{model_version}.csv",
        index=False,
    )
    logger.info("[%s] Saved zero-day detail: %s", model_version, zero_detail_path)
    return zero_detail


def run_v22_v23_zero_day_error(
    output_root: Path,
    zero_config: ZeroDayBacktestConfig | None = None,
    model_configs: dict[str, Path] | None = None,
    force_rerun: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run both model versions and save zero-day detail/summary outputs."""
    zero_config = zero_config or ZeroDayBacktestConfig(
        years=DEFAULT_YEARS,
        stages=DEFAULT_STAGES,
    )
    model_configs = model_configs or DEFAULT_MODEL_CONFIGS
    output_root.mkdir(parents=True, exist_ok=True)

    bootstrap = BacktestingFramework(
        config_path=str(model_configs["v2.2"]),
        output_dir=str(output_root / "bootstrap"),
    )
    latest_pull_date = f"{max(zero_config.years)}-12-31"
    weevil_data = bootstrap.data_service.pull_weevil_data(today=latest_pull_date)
    actual_events = normalize_actual_events(
        weevil_data,
        years=zero_config.years,
        stages=zero_config.stages,
    )
    prediction_dates_by_year, locations_by_year, schedule = build_schedule(
        actual_events,
        zero_config,
    )
    schedule.to_csv(output_root / "zero_day_backtest_schedule.csv", index=False)
    actual_events.to_csv(output_root / "zero_day_actual_events.csv", index=False)

    detail_parts = []
    for model_version, config_path in model_configs.items():
        detail_parts.append(
            run_model_zero_day_backtest(
                model_version=model_version,
                config_path=config_path,
                output_root=output_root,
                weevil_data=weevil_data,
                actual_events=actual_events,
                prediction_dates_by_year=prediction_dates_by_year,
                locations_by_year=locations_by_year,
                zero_config=zero_config,
                force_rerun=force_rerun,
            )
        )

    zero_day_detail = pd.concat(detail_parts, ignore_index=True)
    zero_day_summary = summarize_zero_day_errors(zero_day_detail)
    zero_day_comparison = compare_zero_day_versions(zero_day_summary)

    zero_day_detail.to_parquet(output_root / "zero_day_error_detail_v22_v23.parquet", index=False)
    zero_day_detail.to_csv(output_root / "zero_day_error_detail_v22_v23.csv", index=False)
    zero_day_summary.to_csv(output_root / "zero_day_error_summary_year_stage.csv", index=False)
    zero_day_comparison.to_csv(output_root / "zero_day_error_comparison_v23_minus_v22.csv", index=False)

    return {
        "schedule": schedule,
        "actual_events": actual_events,
        "zero_day_detail": zero_day_detail,
        "zero_day_summary": zero_day_summary,
        "zero_day_comparison": zero_day_comparison,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run v2.2/v2.3 trajectory backtesting and compute zero-day error."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "backtesting_results" / "zero_day_error_v22_v23_3day",
    )
    parser.add_argument("--year", dest="years", action="append", help="Year(s), repeat or comma-separate.")
    parser.add_argument("--stage", dest="stages", action="append", help="Stage(s), repeat or comma-separate.")
    parser.add_argument("--prediction-interval-days", type=int, default=3)
    parser.add_argument("--season-start-md", default="03-01")
    parser.add_argument("--season-end-md", default="06-01")
    parser.add_argument("--exclude-actual-stage-dates", action="store_true")
    parser.add_argument("--max-locations-per-year", type=int)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--zero-threshold", type=float, default=0.0)
    parser.add_argument("--training-window-years", type=int)
    parser.add_argument("--force-rerun", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    zero_config = ZeroDayBacktestConfig(
        years=_parse_int_list(args.years, DEFAULT_YEARS),
        stages=_parse_int_list(args.stages, DEFAULT_STAGES),
        prediction_interval_days=args.prediction_interval_days,
        season_start_md=args.season_start_md,
        season_end_md=args.season_end_md,
        include_actual_stage_dates=not args.exclude_actual_stage_dates,
        max_locations_per_year=args.max_locations_per_year,
        random_seed=args.random_seed,
        zero_threshold=args.zero_threshold,
        training_window_years=args.training_window_years,
    )
    outputs = run_v22_v23_zero_day_error(
        output_root=args.output_root,
        zero_config=zero_config,
        force_rerun=args.force_rerun,
    )
    print(f"OUTPUT_ROOT={args.output_root}")
    print(f"DETAIL_ROWS={len(outputs['zero_day_detail'])}")
    print(f"SUMMARY_ROWS={len(outputs['zero_day_summary'])}")
    print(outputs["zero_day_summary"].to_string(index=False))


if __name__ == "__main__":
    main()

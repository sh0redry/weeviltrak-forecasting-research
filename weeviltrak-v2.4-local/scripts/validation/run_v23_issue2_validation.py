from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "outputs/cache/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline.train_predict import WeevilTrakPipeline
from app.settings import setup_logger

logger = setup_logger()

US_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
}

LAT_BAND_SPECS = [
    (33.0, 35.0, "33-35N"),
    (35.0, 37.0, "35-37N"),
    (37.0, 39.0, "37-39N"),
    (39.0, 40.0, "39-40N"),
    (40.0, 42.0, "40-42N"),
]


def _parse_state_filter_arg(value: str | None) -> set[str] | None:
    if value is None:
        return None
    states = {item.strip().upper() for item in value.split(",") if item.strip()}
    return states or None


def _deduplicate_training_weather(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    return (
        out.sort_values(["location_id", "date"])
        .drop_duplicates(subset=["date", "location_id"], keep="first")
        .copy()
    )


def _assign_lat_band(latitude: float) -> str | None:
    if pd.isna(latitude):
        return None
    for lower, upper, label in LAT_BAND_SPECS:
        if lower <= float(latitude) < upper:
            return label
    return None


def _load_filtered_spatial_coverage(
    pipeline: WeevilTrakPipeline,
    states: set[str] | None = None,
    max_place_ids: int | None = None,
) -> pd.DataFrame:
    spatial_coverage_path = pipeline.config.config["data_source"]["spatial_coverage"]
    coverage = pipeline.s3_manager.read_file(spatial_coverage_path).copy()

    coverage["place_id"] = coverage["place_id"].astype(str)
    coverage["state_code"] = coverage["state_code"].astype(str)
    coverage["centroid_lat"] = pd.to_numeric(coverage["centroid_lat"], errors="coerce")
    coverage["centroid_lon"] = pd.to_numeric(coverage["centroid_lon"], errors="coerce")

    state_filter = states or US_STATE_CODES
    filtered = coverage[
        coverage["state_code"].isin(state_filter) & (coverage["centroid_lat"] < 45.0)
    ].copy()
    filtered["lat_band"] = filtered["centroid_lat"].apply(_assign_lat_band)
    filtered = filtered.dropna(subset=["lat_band"])
    filtered = filtered.drop_duplicates(subset=["place_id"]).reset_index(drop=True)

    if max_place_ids is not None:
        filtered = filtered.head(max_place_ids).copy()

    logger.info(
        "Filtered spatial coverage for issue2 validation: %d rows / %d unique place_id",
        len(filtered),
        filtered["place_id"].nunique(),
    )
    return filtered


def _train_model(
    pipeline: WeevilTrakPipeline,
    training_cutoff_date: str,
) -> None:
    training_today = (pd.to_datetime(training_cutoff_date) - pd.Timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )
    logger.info("Training issue2 validation model with stage_date < %s", training_cutoff_date)

    weevil_train = pipeline.data_service.pull_weevil_data(today=training_today)
    weather_train = pipeline.data_service.query_weather_data_for_weeviltrak_locs(
        config=pipeline.config,
        end_date=training_cutoff_date,
        weevil_data=weevil_train,
    )
    processed_weather = pipeline.data_service.process_weather_data(
        weather_train,
        config=pipeline.config,
        weevil_data=weevil_train,
        mode="train",
    )
    processed_weather_unique = _deduplicate_training_weather(processed_weather)

    x_train, y_train = pipeline.data_service.prepare_training_data(
        weevil_data=weevil_train,
        weather_data=processed_weather_unique,
        test_date=training_cutoff_date,
        features=pipeline.config.features,
        target=pipeline.config.target,
        post_event_training_days=int(
            pipeline.config.config.get(
                "post_event_training_days",
                pipeline.config.config.get("post_event_zero_days", 5),
            )
        ),
    )
    pipeline.model_manager.train(x_train=x_train, y_train=y_train)


def _prepare_prediction_cache(
    pipeline: WeevilTrakPipeline,
    coverage: pd.DataFrame,
    start_date: str,
    end_date: str,
    cache_root: Path,
) -> Path:
    raw_dir = cache_root / "raw_weather"
    processed_dir = cache_root / "processed_weather"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    coverage_enrich = coverage[
        [
            "centroid_lat",
            "centroid_lon",
            "place_id",
            "location_id",
            "latitude",
            "longitude",
        ]
    ].copy()
    coverage_min = coverage_enrich[["centroid_lat", "centroid_lon"]].copy()

    pipeline._cache_raw_weather_two_day_chunks(
        coverage_enrich=coverage_enrich,
        coverage_min=coverage_min,
        pull_start=start_date,
        pull_end=end_date,
        cache_dir=str(raw_dir),
        resolution=pipeline.config.config.get("resolution", "10by10"),
    )
    pipeline._process_weather_batches_from_cache(
        coverage_enrich=coverage_enrich,
        cache_root=str(raw_dir),
        out_dir=str(processed_dir),
        batch_size=300,
    )
    return processed_dir


def _generate_predictions(
    pipeline: WeevilTrakPipeline,
    coverage: pd.DataFrame,
    processed_dir: Path,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    coverage_meta = coverage[
        [
            "place_id",
            "state_code",
            "state_name",
            "centroid_lat",
            "centroid_lon",
            "lat_band",
        ]
    ].drop_duplicates(subset=["place_id"])

    rows = []
    for prediction_date in pd.date_range(start_date, end_date, freq="D"):
        day_str = prediction_date.strftime("%Y-%m-%d")
        x_test = pipeline._build_x_test_for_date(str(processed_dir), day_str)
        if x_test.empty:
            logger.warning("No x_test rows for %s", day_str)
            continue

        x_test = x_test[x_test["stage_id"] == 1].copy()
        if x_test.empty:
            logger.warning("No Stage 1 rows for %s", day_str)
            continue

        y_pred = pipeline.model_manager.predict(x_test[pipeline.config.features])
        out = pipeline._format_prediction_output(x_test=x_test, y_pred=y_pred, today=day_str)
        out = out.merge(coverage_meta, on="place_id", how="left")
        out = out.dropna(subset=["lat_band"]).copy()
        rows.append(out)

    if not rows:
        return pd.DataFrame()

    detail = pd.concat(rows, ignore_index=True)
    detail["prediction_date"] = pd.to_datetime(detail["prediction_date"])
    detail["predicted_stage_date"] = pd.to_datetime(detail["predicted_stage_date"])
    detail["days_until_predicted_stage"] = (
        detail["predicted_stage_date"] - detail["prediction_date"]
    ).dt.days
    return detail


def _build_lat_band_daily_summary(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()

    summary = (
        detail.groupby(["lat_band", "prediction_date"], as_index=False)
        .agg(
            mean_predicted_days=("predicted_days", "mean"),
            median_predicted_days=("predicted_days", "median"),
            mean_raw_signed_days=("raw_signed_days", "mean"),
            mean_predicted_stage_day=("days_until_predicted_stage", "mean"),
            n_places=("place_id", "nunique"),
        )
        .sort_values(["lat_band", "prediction_date"])
    )
    summary["predicted_days_diff"] = (
        summary.groupby("lat_band")["mean_predicted_days"].diff()
    )
    return summary


def _build_lat_band_metric_summary(lat_band_daily: pd.DataFrame) -> pd.DataFrame:
    if lat_band_daily.empty:
        return pd.DataFrame()

    cold_snap_start = pd.Timestamp("2026-03-07")
    cold_snap_end = pd.Timestamp("2026-03-14")
    rows = []
    for lat_band, group in lat_band_daily.groupby("lat_band", sort=False):
        group = group.sort_values("prediction_date").reset_index(drop=True)
        cold_snap = group[
            (group["prediction_date"] >= cold_snap_start)
            & (group["prediction_date"] <= cold_snap_end)
        ].copy()

        max_upward_swing = 0.0
        if not cold_snap.empty:
            diffs = cold_snap["mean_predicted_days"].diff().fillna(0)
            max_upward_swing = float(diffs.clip(lower=0).max())

        first_row = group.iloc[0]
        last_row = group.iloc[-1]
        net_drift_days = float(
            last_row["mean_predicted_stage_day"] - first_row["mean_predicted_stage_day"]
        )
        positive_jump_days = int((group["predicted_days_diff"].fillna(0) > 0).sum())

        rows.append(
            {
                "lat_band": lat_band,
                "max_upward_swing_cold_snap_days": max_upward_swing,
                "net_drift_days": net_drift_days,
                "positive_jump_days": positive_jump_days,
                "start_mean_predicted_days": float(first_row["mean_predicted_days"]),
                "end_mean_predicted_days": float(last_row["mean_predicted_days"]),
                "n_places_end": int(last_row["n_places"]),
            }
        )

    return pd.DataFrame(rows).sort_values("lat_band").reset_index(drop=True)


def _plot_drift_reversal(lat_band_daily: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))

    for lat_band, group in lat_band_daily.groupby("lat_band", sort=False):
        ax.plot(
            group["prediction_date"],
            group["mean_predicted_days"],
            linewidth=2.2,
            label=lat_band,
        )

    ax.set_title("Issue 2 Validation: Stage 1 Forecast Drift Reversal (v2.3)")
    ax.set_xlabel("Prediction Date")
    ax.set_ylabel("Mean Predicted Days")
    ax.axvspan(
        pd.Timestamp("2026-03-07"),
        pd.Timestamp("2026-03-14"),
        color="#d9e8ff",
        alpha=0.4,
    )
    ax.legend(title="Latitude Band")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_summary(
    detail: pd.DataFrame,
    lat_band_metrics: pd.DataFrame,
    output_path: Path,
    figure_path: Path,
    detail_csv_path: Path,
    lat_band_summary_path: Path,
) -> None:
    lines = [
        "# v2.3 Issue 2 Validation Summary",
        "",
        "## Scope",
        "",
        "- Config: `app/config/weeviltrak_v2.3.yml`",
        "- Stage: `Stage 1` only",
        "- Training cutoff: `stage_date < 2026-03-01`",
        "- Prediction window: `2026-03-01` to `2026-03-24`",
        "- Geographic filter: US state whitelist, `centroid_lat < 45`, issue2 latitude bands only",
        "",
        "## Artifacts",
        "",
        f"- Figure: `{figure_path}`",
        f"- Detail CSV: `{detail_csv_path}`",
        f"- Latitude-band summary CSV: `{lat_band_summary_path}`",
        "",
    ]

    if detail.empty or lat_band_metrics.empty:
        lines.extend(
            [
                "## Result",
                "",
                "No validation outputs were generated.",
            ]
        )
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.extend(
        [
            "## Headline",
            "",
            "This validation is intended to compare `v2.3` against the historical Issue 2 drift-reversal baseline represented by `issue2_forecast_drift_reversal.png`.",
            "",
            "## Quantitative Summary",
            "",
            lat_band_metrics.to_markdown(index=False),
            "",
            "## Interpretation Checklist",
            "",
            "- Compare `max_upward_swing_cold_snap_days` against the prior Issue 2 baseline by latitude band.",
            "- Compare `net_drift_days` against the historical directional drift reported in `docs/investigations/stage1_prediction_issues.md`.",
            "- Confirm `positive_jump_days` decreases or at least does not worsen materially in the most affected latitude bands.",
            "- Sanity check the figure for pathological flattening or premature zeroing.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v2.3 Issue 2 validation.")
    parser.add_argument(
        "--config",
        default="app/config/weeviltrak_v2.3.yml",
        help="Config path.",
    )
    parser.add_argument(
        "--training-cutoff-date",
        default="2026-03-01",
        help="Train using stage_date strictly before this date.",
    )
    parser.add_argument(
        "--start-date",
        default="2026-03-01",
        help="Prediction window start date.",
    )
    parser.add_argument(
        "--end-date",
        default="2026-03-24",
        help="Prediction window end date.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/implemention_results",
        help="Directory for generated outputs.",
    )
    parser.add_argument(
        "--states",
        default=None,
        help="Optional comma-separated state_code filter, e.g. KY,WV,NC.",
    )
    parser.add_argument(
        "--max-place-ids",
        type=int,
        default=None,
        help="Optional limit on number of filtered place_id rows for faster debugging.",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep intermediate weather cache directories.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "v2_3_issue2_drift_reversal.png"
    detail_csv_path = output_dir / "v2_3_issue2_detail.csv"
    lat_band_summary_path = output_dir / "v2_3_issue2_lat_band_summary.csv"
    summary_path = output_dir / "v2_3_issue2_validation_summary.md"
    cache_root = Path("outputs/cache/v2_3_issue2_validation")

    pipeline = WeevilTrakPipeline(config_path=args.config)
    coverage = _load_filtered_spatial_coverage(
        pipeline,
        states=_parse_state_filter_arg(args.states),
        max_place_ids=args.max_place_ids,
    )

    try:
        _train_model(
            pipeline=pipeline,
            training_cutoff_date=args.training_cutoff_date,
        )
        processed_dir = _prepare_prediction_cache(
            pipeline=pipeline,
            coverage=coverage,
            start_date=args.start_date,
            end_date=args.end_date,
            cache_root=cache_root,
        )
        detail = _generate_predictions(
            pipeline=pipeline,
            coverage=coverage,
            processed_dir=processed_dir,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        lat_band_daily = _build_lat_band_daily_summary(detail)
        lat_band_metrics = _build_lat_band_metric_summary(lat_band_daily)

        detail.to_csv(detail_csv_path, index=False)
        lat_band_metrics.to_csv(lat_band_summary_path, index=False)
        _plot_drift_reversal(lat_band_daily, figure_path)
        _write_summary(
            detail=detail,
            lat_band_metrics=lat_band_metrics,
            output_path=summary_path,
            figure_path=figure_path,
            detail_csv_path=detail_csv_path,
            lat_band_summary_path=lat_band_summary_path,
        )
        logger.info("Saved v2.3 Issue 2 validation outputs under %s", output_dir)
    finally:
        if not args.keep_cache and cache_root.exists():
            shutil.rmtree(cache_root, ignore_errors=True)


if __name__ == "__main__":
    main()

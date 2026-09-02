from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline.backtesting import BacktestingFramework


def _classify_root_cause(row: pd.Series) -> str:
    early_gap = pd.to_numeric(row.get("days_earlier_than_history_median"), errors="coerce")
    day0_pred = pd.to_numeric(row.get("day0_predicted_days"), errors="coerce")
    min_pred = pd.to_numeric(row.get("min_predicted_days"), errors="coerce")
    cumu_gdd = pd.to_numeric(row.get("day0_cumu_gdd_air"), errors="coerce")
    latitude = pd.to_numeric(row.get("day0_latitude"), errors="coerce")

    if pd.notna(early_gap) and early_gap >= 14 and pd.notna(latitude) and latitude <= 39.0:
        return "historically_early_likely_training_coverage_gap"
    if pd.notna(min_pred) and min_pred <= 3:
        return "near_convergence_likely_postprocess_or_short_window"
    if pd.notna(cumu_gdd) and cumu_gdd >= 50 and pd.notna(day0_pred) and day0_pred >= 7:
        return "warm_signal_but_late_prediction_possible_feature_or_generalization_gap"
    return "requires_manual_review"


def build_non_converged_summary(
    validation_csv: Path,
    output_dir: Path,
    config_path: str,
) -> tuple[Path, Path]:
    df = pd.read_csv(validation_csv)
    df["location_id"] = df["location_id"].astype(str)
    df["stage_date"] = pd.to_datetime(df["stage_date"])
    df["prediction_date"] = pd.to_datetime(df["prediction_date"])
    df["actual_stage_date"] = pd.to_datetime(df["actual_stage_date"])

    window_summary = (
        df.groupby(["event_year", "location_id", "stage_date"], as_index=False)
        .agg(
            min_predicted_days=("predicted_days", "min"),
            max_predicted_days=("predicted_days", "max"),
            median_abs_error=("abs_error_days", "median"),
            mean_abs_error=("abs_error_days", "mean"),
            terminated_within_window=("predicted_days", lambda s: (pd.to_numeric(s, errors="coerce") <= 0).any()),
        )
    )

    day0 = (
        df[df["days_from_event"] == 0][
            ["event_year", "location_id", "stage_date", "predicted_days"]
        ]
        .rename(columns={"predicted_days": "day0_predicted_days"})
        .copy()
    )
    window_summary = window_summary.merge(
        day0,
        on=["event_year", "location_id", "stage_date"],
        how="left",
    )

    non_converged = window_summary[~window_summary["terminated_within_window"]].copy()
    if non_converged.empty:
        detail_path = output_dir / "issue1_non_converged_windows.csv"
        summary_path = output_dir / "issue1_non_converged_windows_summary.md"
        non_converged.to_csv(detail_path, index=False)
        summary_path.write_text("# Issue 1 Non-Converged Window Analysis\n\nNo non-converged windows found.\n", encoding="utf-8")
        return detail_path, summary_path

    framework = BacktestingFramework(config_path=config_path, output_dir=str(output_dir / "tmp_issue1_non_converged"))
    weevil_data = framework.data_service.pull_weevil_data(today="2025-12-31")
    weevil_data["location_id"] = weevil_data["location_id"].astype(str)
    weevil_data["stage_id"] = pd.to_numeric(weevil_data["stage_id"], errors="coerce")
    weevil_data["stage_date"] = pd.to_datetime(weevil_data["stage_date"])

    # Historical Stage 1 timing by location.
    hist_stage1 = weevil_data[weevil_data["stage_id"] == 1].copy()
    hist_stage1["event_year"] = hist_stage1["stage_date"].dt.year
    hist_stage1["history_doy"] = hist_stage1["stage_date"].dt.dayofyear

    hist_parts = []
    for row in non_converged.itertuples(index=False):
        location_hist = hist_stage1[
            (hist_stage1["location_id"] == row.location_id)
            & (hist_stage1["event_year"] != int(row.event_year))
        ].copy()
        history_years = sorted(location_hist["event_year"].dropna().astype(int).unique().tolist())
        hist_parts.append(
            {
                "event_year": int(row.event_year),
                "location_id": str(row.location_id),
                "stage_date": pd.to_datetime(row.stage_date),
                "current_doy": int(pd.to_datetime(row.stage_date).dayofyear),
                "history_n": int(len(location_hist)),
                "history_years": ",".join(map(str, history_years)),
                "history_median_doy": location_hist["history_doy"].median() if not location_hist.empty else pd.NA,
                "history_min_doy": location_hist["history_doy"].min() if not location_hist.empty else pd.NA,
                "history_max_doy": location_hist["history_doy"].max() if not location_hist.empty else pd.NA,
            }
        )
    hist_df = pd.DataFrame(hist_parts)
    non_converged = non_converged.merge(
        hist_df,
        on=["event_year", "location_id", "stage_date"],
        how="left",
    )
    non_converged["days_earlier_than_history_median"] = (
        pd.to_numeric(non_converged["history_median_doy"], errors="coerce")
        - pd.to_numeric(non_converged["current_doy"], errors="coerce")
    )

    # Pull event-day features for the non-converged locations only.
    loc_ids = sorted(non_converged["location_id"].astype(str).unique().tolist())
    event_dates = sorted(pd.to_datetime(non_converged["stage_date"]).dt.date.unique().tolist())
    weevil_subset = weevil_data[weevil_data["location_id"].isin(loc_ids)].copy()
    processed_weather = framework.prepare_weather(weevil_subset, event_dates)
    processed_weather["location_id"] = processed_weather["location_id"].astype(str)
    processed_weather["date"] = pd.to_datetime(processed_weather["date"])

    day0_features = processed_weather[
        processed_weather["date"].isin(pd.to_datetime(non_converged["stage_date"]).dt.normalize())
    ][
        [
            "location_id",
            "date",
            "cumu_gdd_air",
            "rolling_gdd_air",
            "rolling_humidity_mean_pct",
            "cumu_precip_total_mm",
            "latitude",
            "longitude",
        ]
    ].copy()
    day0_features = day0_features.rename(
        columns={
            "date": "stage_date",
            "cumu_gdd_air": "day0_cumu_gdd_air",
            "rolling_gdd_air": "day0_rolling_gdd_air",
            "rolling_humidity_mean_pct": "day0_rolling_humidity_mean_pct",
            "cumu_precip_total_mm": "day0_cumu_precip_total_mm",
            "latitude": "day0_latitude",
            "longitude": "day0_longitude",
        }
    )
    non_converged = non_converged.merge(
        day0_features,
        on=["location_id", "stage_date"],
        how="left",
    )
    non_converged["root_cause_label"] = non_converged.apply(_classify_root_cause, axis=1)

    non_converged = non_converged.sort_values(
        ["event_year", "median_abs_error", "location_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    detail_path = output_dir / "issue1_non_converged_windows.csv"
    non_converged.to_csv(detail_path, index=False)

    summary_lines = [
        "# Issue 1 Non-Converged Window Analysis",
        "",
        "## Summary",
        "",
        f"- Non-converged windows: `{len(non_converged)}`",
        f"- Distinct location_id: `{non_converged['location_id'].nunique()}`",
        "",
        "## Root Cause Buckets",
        "",
    ]
    bucket_counts = (
        non_converged["root_cause_label"]
        .value_counts(dropna=False)
        .rename_axis("root_cause_label")
        .reset_index(name="count")
    )
    for row in bucket_counts.itertuples(index=False):
        summary_lines.append(f"- `{row.root_cause_label}`: `{row.count}`")

    summary_lines.extend(["", "## Window Detail", ""])
    for row in non_converged.itertuples(index=False):
        summary_lines.extend(
            [
                f"### location_id {row.location_id} / {row.event_year}",
                "",
                f"- stage_date: `{pd.to_datetime(row.stage_date).date()}`",
                f"- min_predicted_days: `{row.min_predicted_days:.3f}`",
                f"- day0_predicted_days: `{row.day0_predicted_days:.3f}`",
                f"- median_abs_error: `{row.median_abs_error:.3f}`",
                f"- history_years: `{row.history_years}`",
                f"- history_median_doy: `{row.history_median_doy}`",
                f"- current_doy: `{row.current_doy}`",
                f"- days_earlier_than_history_median: `{row.days_earlier_than_history_median}`",
                f"- day0_cumu_gdd_air: `{row.day0_cumu_gdd_air}`",
                f"- day0_rolling_gdd_air: `{row.day0_rolling_gdd_air}`",
                f"- day0_latitude/day0_longitude: `{row.day0_latitude}, {row.day0_longitude}`",
                f"- root_cause_label: `{row.root_cause_label}`",
                "",
            ]
        )

    summary_path = output_dir / "issue1_non_converged_windows_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    return detail_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze non-converged Issue 1 validation windows.")
    parser.add_argument(
        "--validation-csv",
        default="docs/implemention_results/issue1_signed_target_validation.csv",
        help="Validation CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/implemention_results",
        help="Output directory for non-converged analysis.",
    )
    parser.add_argument(
        "--config",
        default="app/config/weeviltrak_v2.3.yml",
        help="Config path.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path, summary_path = build_non_converged_summary(
        validation_csv=Path(args.validation_csv),
        output_dir=output_dir,
        config_path=args.config,
    )
    print(f"Saved detail to {detail_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()

"""
Lightweight business check for monotonic constraints.

Compares v2.1 Random Forest vs the current LightGBM config on real weevil event windows:
  - stage_id fixed to 2 by default
  - prediction windows from stage_date - 7 to stage_date + 2
  - train once per model, predict many dates/locations
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.pipeline.backtesting import BacktestingFramework
from app.settings import setup_logger

logger = setup_logger()


def build_event_windows(
    weevil_data: pd.DataFrame,
    stage_id: int = 2,
    days_before: int = 7,
    days_after: int = 2,
) -> pd.DataFrame:
    """
    Expand each real event into one row per prediction date in the inspection window.
    """
    required = {"location_id", "stage_id", "stage_date", "latitude", "longitude"}
    missing = required - set(weevil_data.columns)
    if missing:
        raise ValueError(f"weevil_data missing required columns: {sorted(missing)}")

    events = weevil_data.copy()
    events["stage_date"] = pd.to_datetime(events["stage_date"]).dt.normalize()
    events["location_id"] = events["location_id"].astype(str)
    events["stage_id"] = pd.to_numeric(events["stage_id"], errors="coerce").astype(int)

    events = events[events["stage_id"] == stage_id].copy()
    if events.empty:
        raise ValueError(f"No events found for stage_id={stage_id}")

    rows: List[Dict[str, object]] = []
    for row in events.itertuples(index=False):
        stage_date = pd.to_datetime(row.stage_date).normalize()
        for prediction_date in pd.date_range(
            stage_date - pd.Timedelta(days=days_before),
            stage_date + pd.Timedelta(days=days_after),
            freq="D",
        ):
            rows.append(
                {
                    "location_id": str(row.location_id),
                    "stage_id": int(row.stage_id),
                    "stage_date": stage_date,
                    "prediction_date": pd.to_datetime(prediction_date).normalize(),
                    "latitude": float(row.latitude),
                    "longitude": float(row.longitude),
                }
            )

    windows = pd.DataFrame(rows)
    windows = windows.drop_duplicates(
        subset=["location_id", "stage_id", "stage_date", "prediction_date"],
        keep="first",
    ).copy()
    windows["days_from_event"] = (
        windows["prediction_date"] - windows["stage_date"]
    ).dt.days
    windows = windows.sort_values(
        ["location_id", "stage_date", "prediction_date"]
    ).reset_index(drop=True)
    return windows


def summarize_behavior(detail_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise bounce-back and post-event-positive behavior by model_version.
    """
    if detail_df.empty:
        return pd.DataFrame(
            columns=[
                "model_version",
                "n_event_windows",
                "n_prediction_rows",
                "bounce_back_count",
                "bounce_back_rate",
                "post_event_positive_count",
                "post_event_positive_rate",
            ]
        )

    df = detail_df.copy()
    df = df.dropna(subset=["model_version", "predicted_days"]).copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "model_version",
                "n_event_windows",
                "n_prediction_rows",
                "bounce_back_count",
                "bounce_back_rate",
                "post_event_positive_count",
                "post_event_positive_rate",
            ]
        )

    df["prediction_date"] = pd.to_datetime(df["prediction_date"]).dt.normalize()
    df["stage_date"] = pd.to_datetime(df["stage_date"]).dt.normalize()
    df["location_id"] = df["location_id"].astype(str)
    df["stage_id"] = pd.to_numeric(df["stage_id"], errors="coerce").astype(int)
    df = df.sort_values(
        ["model_version", "location_id", "stage_id", "stage_date", "prediction_date"]
    )

    group_cols = ["model_version", "location_id", "stage_id", "stage_date"]
    df["predicted_days_next"] = df.groupby(group_cols)["predicted_days"].shift(-1)
    bounce_back_threshold = 1.0
    df["is_bounce_back"] = (
        df["predicted_days_next"].notna()
        & ((df["predicted_days_next"] - df["predicted_days"]) > bounce_back_threshold)
    )
    df["is_post_event_positive"] = (
        (df["prediction_date"] > df["stage_date"]) & (df["predicted_days"] > 0)
    )

    summaries: List[Dict[str, object]] = []
    for model_version, g in df.groupby("model_version", dropna=False):
        n_event_windows = (
            g[["location_id", "stage_id", "stage_date"]].drop_duplicates().shape[0]
        )
        n_prediction_rows = len(g)
        n_transitions = int(g["predicted_days_next"].notna().sum())
        n_post_event_rows = int((g["prediction_date"] > g["stage_date"]).sum())
        bounce_back_count = int(g["is_bounce_back"].sum())
        post_event_positive_count = int(g["is_post_event_positive"].sum())

        summaries.append(
            {
                "model_version": model_version,
                "n_event_windows": n_event_windows,
                "n_prediction_rows": n_prediction_rows,
                "bounce_back_count": bounce_back_count,
                "bounce_back_rate": (
                    bounce_back_count / n_transitions if n_transitions else np.nan
                ),
                "post_event_positive_count": post_event_positive_count,
                "post_event_positive_rate": (
                    post_event_positive_count / n_post_event_rows
                    if n_post_event_rows
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(summaries).sort_values("model_version").reset_index(drop=True)


def apply_zero_day_termination(
    detail_df: pd.DataFrame,
    threshold: float = 1.0,
    consecutive_days: int = 2,
) -> pd.DataFrame:
    """
    Lock a trajectory to zero after the first consecutive near-zero run.
    """
    if detail_df.empty:
        return detail_df
    if consecutive_days <= 1:
        raise ValueError("consecutive_days must be >= 2")

    df = detail_df.copy()
    df["prediction_date"] = pd.to_datetime(df["prediction_date"]).dt.normalize()
    df["stage_date"] = pd.to_datetime(df["stage_date"]).dt.normalize()
    df["predicted_stage_date"] = pd.to_datetime(
        df["predicted_stage_date"], errors="coerce"
    ).dt.normalize()
    df = df.sort_values(["location_id", "stage_id", "stage_date", "prediction_date"])

    out_parts: List[pd.DataFrame] = []
    group_cols = ["location_id", "stage_id", "stage_date"]
    for _, g in df.groupby(group_cols, dropna=False, sort=False):
        g = g.copy().sort_values("prediction_date").reset_index(drop=True)
        near_zero = pd.to_numeric(g["predicted_days"], errors="coerce") <= threshold

        lock_idx = None
        run = 0
        for i, is_near_zero in enumerate(near_zero.tolist()):
            run = run + 1 if is_near_zero else 0
            if run >= consecutive_days:
                lock_idx = i - consecutive_days + 1
                break

        if lock_idx is not None:
            locked_event_date = pd.to_datetime(g.loc[lock_idx, "prediction_date"]).normalize()
            g.loc[lock_idx:, "predicted_days"] = 0.0
            g.loc[lock_idx:, "predicted_stage_date"] = locked_event_date

        out_parts.append(g)

    return pd.concat(out_parts, ignore_index=True)


def _normalize_detail_for_output(detail_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize date-like columns to stable parquet-friendly string dates.
    """
    if detail_df.empty:
        return detail_df

    df = detail_df.copy()
    for col in ["stage_date", "prediction_date", "predicted_stage_date"]:
        if col in df.columns:
            dt = pd.to_datetime(df[col], errors="coerce")
            df[col] = dt.dt.strftime("%Y-%m-%d")
            df.loc[dt.isna(), col] = pd.NA
    return df


class MonotonicBusinessCheck:
    """
    Lightweight A/B comparison between RF v2.1 and constrained LightGBM.
    """

    def __init__(
        self,
        rf_config_path: str = "app/config/weeviltrak_v2.1.yml",
        lgbm_config_path: str = "app/config/weeviltrak_v2.3.yml",
        output_dir: Optional[str] = None,
    ):
        self.rf_framework = BacktestingFramework(
            config_path=rf_config_path,
            output_dir=output_dir or "outputs/business_checks/v2.1",
        )
        self.lgbm_framework = BacktestingFramework(
            config_path=lgbm_config_path,
            output_dir=output_dir or "outputs/business_checks/v2.3",
        )
        self.output_dir = Path(output_dir or "outputs/business_checks")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.zero_day_threshold = float(
            self.lgbm_framework.config.config.get("zero_day_termination_threshold", 1.0)
        )
        self.zero_day_consecutive_days = int(
            self.lgbm_framework.config.config.get(
                "zero_day_termination_consecutive_days", 2
            )
        )

        if self.rf_framework.config.features != self.lgbm_framework.config.features:
            raise ValueError(
                "RF and LightGBM configs must use the same features for aligned comparison."
            )

    def _predict_for_event_windows(
        self,
        framework: BacktestingFramework,
        model,
        event_windows: pd.DataFrame,
        processed_weather_unique: pd.DataFrame,
        model_version: str,
        cutoff_date: str,
    ) -> pd.DataFrame:
        prediction_dates = sorted(
            pd.to_datetime(event_windows["prediction_date"]).dt.date.unique()
        )

        pred_parts: List[pd.DataFrame] = []
        cutoff_str = pd.to_datetime(cutoff_date).strftime("%Y-%m-%d")

        for d in prediction_dates:
            locations_for_date = sorted(
                event_windows.loc[
                    pd.to_datetime(event_windows["prediction_date"]).dt.date == d,
                    "location_id",
                ]
                .astype(str)
                .unique()
                .tolist()
            )
            if not locations_for_date:
                continue

            out_df = framework.predict_with_model_for_date(
                model=model,
                test_date=d,
                locations_for_date=locations_for_date,
                model_train_cutoff_date=cutoff_str,
                processed_weather_unique=processed_weather_unique,
            )
            if out_df is None or out_df.empty:
                continue

            key_cols = ["prediction_date", "location_id", "stage_id"]
            if out_df.duplicated(subset=key_cols, keep=False).any():
                out_df = framework.aggregate_predictions_unique(out_df)

            out_df = out_df.copy()
            out_df["prediction_date"] = pd.to_datetime(out_df["prediction_date"]).dt.normalize()
            out_df["location_id"] = out_df["location_id"].astype(str)
            out_df["stage_id"] = pd.to_numeric(out_df["stage_id"], errors="coerce").astype(int)
            pred_parts.append(out_df)

        if not pred_parts:
            return pd.DataFrame()

        preds = pd.concat(pred_parts, ignore_index=True)
        preds["model_version"] = model_version

        detail = event_windows.merge(
            preds[
                [
                    "model_version",
                    "location_id",
                    "stage_id",
                    "prediction_date",
                    "predicted_days",
                    "predicted_stage_date",
                ]
            ],
            on=["location_id", "stage_id", "prediction_date"],
            how="left",
        )
        detail["model_train_cutoff_date"] = cutoff_str
        return detail

    def run(
        self,
        today: str,
        cutoff_date: Optional[str] = None,
        stage_id: int = 2,
        days_before: int = 7,
        days_after: int = 2,
    ) -> Dict[str, str]:
        today_str = pd.to_datetime(today).strftime("%Y-%m-%d")
        cutoff_str = pd.to_datetime(cutoff_date or today_str).strftime("%Y-%m-%d")

        logger.info(
            "Running monotonic business check for today=%s cutoff_date=%s stage_id=%s",
            today_str,
            cutoff_str,
            stage_id,
        )

        weevil_data = self.rf_framework.data_service.pull_weevil_data(today=today_str)
        event_windows = build_event_windows(
            weevil_data=weevil_data,
            stage_id=stage_id,
            days_before=days_before,
            days_after=days_after,
        )
        logger.info(
            "Built %d event-window rows across %d unique windows",
            len(event_windows),
            event_windows[["location_id", "stage_id", "stage_date"]]
            .drop_duplicates()
            .shape[0],
        )

        processed_weather_unique = self.rf_framework.prepare_weather(
            weevil_data=weevil_data,
            test_dates=sorted(
                pd.to_datetime(event_windows["prediction_date"]).dt.date.unique()
            ),
        )

        rf_model = self.rf_framework.train_model_for_cutoff(
            cutoff_date=cutoff_str,
            processed_weather_unique=processed_weather_unique,
            training_window_years=None,
        )
        lgbm_model = self.lgbm_framework.train_model_for_cutoff(
            cutoff_date=cutoff_str,
            processed_weather_unique=processed_weather_unique,
            training_window_years=None,
        )

        rf_detail = self._predict_for_event_windows(
            framework=self.rf_framework,
            model=rf_model,
            event_windows=event_windows,
            processed_weather_unique=processed_weather_unique,
            model_version=self.rf_framework.config.config.get("version", "v2.1"),
            cutoff_date=cutoff_str,
        )
        lgbm_detail = self._predict_for_event_windows(
            framework=self.lgbm_framework,
            model=lgbm_model,
            event_windows=event_windows,
            processed_weather_unique=processed_weather_unique,
            model_version=self.lgbm_framework.config.config.get("version", "v2.2"),
            cutoff_date=cutoff_str,
        )
        lgbm_detail = apply_zero_day_termination(
            lgbm_detail,
            threshold=self.zero_day_threshold,
            consecutive_days=self.zero_day_consecutive_days,
        )

        detail = pd.concat([rf_detail, lgbm_detail], ignore_index=True)
        detail = detail.sort_values(
            ["model_version", "location_id", "stage_date", "prediction_date"]
        ).reset_index(drop=True)
        detail = _normalize_detail_for_output(detail)
        summary = summarize_behavior(detail)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        detail_csv = self.output_dir / f"monotonic_business_check_detail_{ts}.csv"
        detail_parquet = (
            self.output_dir / f"monotonic_business_check_detail_{ts}.parquet"
        )
        summary_csv = self.output_dir / f"monotonic_business_check_summary_{ts}.csv"

        detail.to_csv(detail_csv, index=False)
        detail.to_parquet(detail_parquet, index=False)
        summary.to_csv(summary_csv, index=False)

        logger.info("Saved detail CSV: %s", detail_csv)
        logger.info("Saved detail Parquet: %s", detail_parquet)
        logger.info("Saved summary CSV: %s", summary_csv)
        logger.info("Summary:\n%s", summary.to_string(index=False))

        return {
            "detail_csv": str(detail_csv),
            "detail_parquet": str(detail_parquet),
            "summary_csv": str(summary_csv),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run a lightweight monotonic-constraints business check."
    )
    parser.add_argument(
        "--today",
        required=True,
        help="Upper bound passed to pull_weevil_data(), e.g. 2025-06-01",
    )
    parser.add_argument(
        "--cutoff-date",
        required=False,
        help="Fixed training cutoff date for both models. Defaults to --today.",
    )
    parser.add_argument(
        "--rf-config-path",
        default="app/config/weeviltrak_v2.1.yml",
        help="RF baseline config path.",
    )
    parser.add_argument(
        "--lgbm-config-path",
        default="app/config/weeviltrak_v2.3.yml",
        help="LightGBM monotonic config path.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/business_checks",
        help="Directory for detail and summary outputs.",
    )
    parser.add_argument(
        "--stage-id",
        type=int,
        default=2,
        help="Stage ID to inspect. Default is 2.",
    )
    args = parser.parse_args()

    checker = MonotonicBusinessCheck(
        rf_config_path=args.rf_config_path,
        lgbm_config_path=args.lgbm_config_path,
        output_dir=args.output_dir,
    )
    checker.run(
        today=args.today,
        cutoff_date=args.cutoff_date,
        stage_id=args.stage_id,
    )


if __name__ == "__main__":
    main()

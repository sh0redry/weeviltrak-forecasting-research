"""
Backtesting module for WeevilTrak model evaluation.

Follows the plan described in Backtesting.md:
  Step 1 — Build backtesting plan (date_to_locations mapping + plan CSV)
  Step 2 — Pull & process weather once, deduplicate to processed_weather_unique
  Step 3 — Walk-forward loop: train once per year -> predict -> save (incremental, with resume)

Production mode (default):
  - Train ONE model per prediction year using the configured training window.
  - Cutoff = Jan 1 of prediction year (e.g. train on 2018–2022 for 2023 predictions).
  - Reuse the same yearly model for all testing_dates within that year.

Outputs:
  predictions_detail_lead{N}_{ts}.tmp.csv   — incremental checkpoint (append-safe)
  predictions_detail_lead{N}_{ts}.parquet   — converted from CSV after run completes
  predictions_detail.parquet                — latest copy (overwritten each run)
  testing_results_year_stage_lead{N}_{ts}.csv  — evaluation by (year × stage)
  testing_results_year_overall_lead{N}_{ts}.csv — evaluation by year
  run_log.csv / run_log_final.csv           — per-date profiling & status
  run_summary.csv                           — aggregated status counts
  failures.csv / failures_final.csv         — error details for failed dates
  backtesting_plan_events.csv               — source of truth for what is being tested
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from app.pipeline.base import PipelineBase
from app.pipeline.termination import (
    LearnedTerminationSettings,
    apply_learned_termination_rule,
    select_aligned_training_window,
    train_learned_termination_model,
)
from app.settings import setup_logger

logger = setup_logger()


class BacktestingFramework(PipelineBase):
    """
    Framework for backtesting the WeevilTrak model across multiple years.

    Key design principles:
    - Production mode: train ONE model per year (configured window, cutoff = Jan 1).
    - testing_date = actual stage_date - lead_days
    - Weather is pulled & processed once, then reused across all dates.
    - Predictions are filtered to only the locations scheduled for each date.
    - Duplicate predictions are aggregated using the median (robust).
    - Results are saved incrementally as CSV then converted to Parquet.
    - Checkpoint / resume support: skip dates already in the output file.
    """

    def __init__(self, config_path: str, output_dir: str = None):
        """
        Initialise the framework and all underlying services.

        Args:
            config_path: Path to YAML configuration file (for example
                ``app/config/weeviltrak_v2.4.yml`` for the current canonical
                Stage/Phase pipeline)
            output_dir: Local directory for CSV / Parquet outputs.
                        If None, derives a versioned directory from config
                        (for example ``outputs/backtesting/v2.3/``).
        """
        super().__init__(config_path)

        if output_dir is None:
            version = self.config.config.get("version", "unknown")
            output_dir = f"outputs/backtesting/{version}"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._termination_models_by_cutoff: Dict[str, object] = {}

        logger.info("BacktestingFramework initialised")

    # ------------------------------------------------------------------
    # Step 1 — Build backtesting plan
    # ------------------------------------------------------------------

    def build_backtesting_plan(
        self,
        years: List[int] = [2023, 2024, 2025],
        stages: List[int] = [1, 4, 2, 3],
        lead_days: int = 7,
    ) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
        """
        Build the backtesting plan:
          - Pull weevil data for the target years
          - Compute testing_date = stage_date - lead_days
          - Build date_to_locations mapping
          - Save backtesting_plan_events.csv

        Args:
            years: Years to include
            stages: Stage IDs to include
            lead_days: Number of days before stage_date to set as testing_date

        Returns:
            (date_to_locations, events DataFrame, full weevil_data DataFrame)
        """
        logger.info(
            f"Step 1: Building backtesting plan for years={years}, stages={stages}"
        )

        max_year = max(years)
        latest_date = datetime(max_year, 12, 31) + timedelta(days=lead_days + 2)

        weevil_data = self.data_service.pull_weevil_data(
            today=latest_date.strftime("%Y-%m-%d")
        )
        weevil_data["stage_date"] = pd.to_datetime(weevil_data["stage_date"])

        # Filter to backtest scope
        events = weevil_data[
            (weevil_data["year"].isin(years)) & (weevil_data["stage_id"].isin(stages))
        ].copy()

        if events.empty:
            raise RuntimeError(
                f"No weevil events found for years={years} and stages={stages}."
            )

        events["testing_date"] = events["stage_date"] - pd.Timedelta(days=lead_days)
        events["testing_date_only"] = events["testing_date"].dt.date

        # Build date -> locations mapping
        date_to_locations: Dict = (
            events.groupby("testing_date_only")["location_id"]
            .apply(lambda s: sorted(s.astype(str).unique().tolist()))
            .to_dict()
        )

        # Save plan CSV
        events["location_id"] = events["location_id"].astype(str)
        events["stage_id"] = pd.to_numeric(events["stage_id"], errors="coerce").astype(
            int
        )

        events_to_save = (
            events[
                [
                    "location_id",
                    "stage_id",
                    "year",
                    "stage_date",
                    "testing_date",
                    "testing_date_only",
                    "latitude",
                    "longitude",
                ]
            ]
            .copy()
            .sort_values(["location_id", "stage_date", "stage_id"])
            .reset_index(drop=True)
        )

        plan_file = self.output_dir / "backtesting_plan_events.csv"
        events_to_save.to_csv(plan_file, index=False)

        logger.info(
            f"Backtesting plan saved: {plan_file}  ({len(events_to_save)} rows)"
        )
        logger.info(f"Unique testing dates: {len(date_to_locations)}")

        return date_to_locations, events, weevil_data

    @staticmethod
    def build_event_window_plan(
        events: pd.DataFrame,
        days_before: int = 3,
        days_after: int = 3,
    ) -> pd.DataFrame:
        """
        Expand each selected backtesting event into an event-centered window.

        The input ``events`` should come from ``build_backtesting_plan`` so this
        validation uses the same years/stages/location scope as normal
        backtesting. Each event becomes one row per prediction date in:

            [stage_date - days_before, stage_date + days_after]
        """
        if events is None or events.empty:
            return pd.DataFrame()
        if days_before < 0 or days_after < 0:
            raise ValueError("days_before and days_after must be non-negative")

        required = [
            "location_id",
            "stage_id",
            "year",
            "stage_date",
            "latitude",
            "longitude",
        ]
        missing = [col for col in required if col not in events.columns]
        if missing:
            raise ValueError(f"build_event_window_plan: missing columns {missing}")

        event_data = events.copy()
        event_data["location_id"] = event_data["location_id"].astype(str)
        event_data["stage_id"] = pd.to_numeric(
            event_data["stage_id"], errors="coerce"
        ).astype(int)
        event_data["stage_date"] = pd.to_datetime(
            event_data["stage_date"]
        ).dt.normalize()

        rows = []
        for row in event_data.itertuples(index=False):
            stage_date = pd.to_datetime(row.stage_date).normalize()
            for prediction_date in pd.date_range(
                stage_date - pd.Timedelta(days=days_before),
                stage_date + pd.Timedelta(days=days_after),
                freq="D",
            ):
                rows.append(
                    {
                        "event_year": int(row.year),
                        "location_id": str(row.location_id),
                        "stage_id": int(row.stage_id),
                        "stage_date": stage_date,
                        "prediction_date": pd.to_datetime(prediction_date).normalize(),
                        "latitude": float(row.latitude),
                        "longitude": float(row.longitude),
                    }
                )

        windows = pd.DataFrame(rows)
        if windows.empty:
            return windows

        windows = windows.drop_duplicates(
            subset=[
                "event_year",
                "location_id",
                "stage_id",
                "stage_date",
                "prediction_date",
            ],
            keep="first",
        ).copy()
        windows["days_from_event"] = (
            windows["prediction_date"] - windows["stage_date"]
        ).dt.days
        windows["prediction_date_only"] = windows["prediction_date"].dt.date
        windows = windows.sort_values(
            ["event_year", "location_id", "stage_id", "stage_date", "prediction_date"]
        ).reset_index(drop=True)
        return windows

    # ------------------------------------------------------------------
    # Step 2 — Pull & process weather once
    # ------------------------------------------------------------------

    def prepare_weather(
        self,
        weevil_data: pd.DataFrame,
        test_dates: List,
    ) -> pd.DataFrame:
        """
        Pull weather data once (up to the latest testing_date) and process it.
        Then deduplicate to at most one row per (date, location_id).

        Args:
            weevil_data: Full weevil DataFrame (needed by process_weather_data)
            test_dates: Sorted list of testing dates

        Returns:
            processed_weather_unique DataFrame
        """
        logger.info("Step 2: Pulling and processing weather data (once)...")

        max_test_date_str = pd.to_datetime(max(test_dates)).strftime("%Y-%m-%d")

        weather_raw = self.data_service.query_weather_data_for_weeviltrak_locs(
            config=self.config,
            end_date=max_test_date_str,
            weevil_data=weevil_data,
        )

        processed_weather = self.data_service.process_weather_data(
            weather_data=weather_raw,
            config=self.config,
            weevil_data=weevil_data,
        )

        # Deduplicate: keep one row per (date, location_id)
        pw = processed_weather.copy()
        pw["date"] = pd.to_datetime(pw["date"])
        pw["location_id"] = pw["location_id"].astype(str)

        rows_before = len(pw)
        pw = (
            pw.sort_values(["location_id", "date"])
            .drop_duplicates(subset=["date", "location_id"], keep="first")
            .copy()
        )
        rows_after = len(pw)

        logger.info(
            f"Weather deduplication: {rows_before} -> {rows_after} rows "
            f"(compression {rows_before / max(rows_after, 1):.2f}x)"
        )

        return pw

    # ------------------------------------------------------------------
    # Step 3a — Production helpers: train once per year
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_predictions_unique(out_df: pd.DataFrame) -> pd.DataFrame:
        """
        Collapse duplicate predictions to 1 row per (prediction_date, location_id, stage_id).
        predicted_days uses the median (robust). Adds n_rows column for diagnostics.

        Args:
            out_df: Raw predictions DataFrame

        Returns:
            Aggregated DataFrame with one row per (prediction_date, location_id, stage_id)
        """
        if out_df is None or out_df.empty:
            return out_df

        req = ["prediction_date", "location_id", "stage_id", "predicted_days"]
        missing = [c for c in req if c not in out_df.columns]
        if missing:
            raise ValueError(f"aggregate_predictions_unique: missing columns {missing}")

        df = out_df.copy()
        df["location_id"] = df["location_id"].astype(str)

        # Carry forward any extra scalar columns that are constant per key
        extra_cols = [
            c
            for c in df.columns
            if c
            not in req
            + ["latitude", "longitude", "predicted_stage_date", "raw_signed_days"]
        ]

        agg_spec: Dict = {
            "latitude": ("latitude", "first"),
            "longitude": ("longitude", "first"),
            "predicted_days": ("predicted_days", "median"),
            "n_rows": ("predicted_days", "size"),
        }
        if "raw_signed_days" in df.columns:
            agg_spec["raw_signed_days"] = ("raw_signed_days", "median")
        for col in extra_cols:
            agg_spec[col] = (col, "first")

        agg = df.groupby(
            ["prediction_date", "location_id", "stage_id"], as_index=False
        ).agg(**agg_spec)

        pred_dt = pd.to_datetime(agg["prediction_date"])
        agg["predicted_stage_date"] = (
            pred_dt + pd.to_timedelta(agg["predicted_days"], unit="D")
        ).dt.strftime("%Y-%m-%d")

        return agg

    def train_model_for_cutoff(
        self,
        cutoff_date,
        processed_weather_unique: pd.DataFrame,
        training_window_years: Optional[int] = None,
        use_config_default: bool = True,
    ):
        """
        Train one model using data strictly before cutoff_date (expanding window by default).

        Yearly schedule (expanding):
          - Predict 2023 → cutoff 2023-01-01 (train on 2018–2022)
          - Predict 2024 → cutoff 2024-01-01 (train on 2018–2023)
          - Predict 2025 → cutoff 2025-01-01 (train on 2018–2024)

        Args:
            cutoff_date: Date-like; training uses records strictly before this date.
            processed_weather_unique: Pre-processed, deduplicated weather DataFrame.
            training_window_years: None uses the config default (expanding when
                absent); int uses a rolling last-N-year window.
            use_config_default: Set false with ``training_window_years=None``
                to explicitly request an expanding window even when the model
                config declares a default rolling window.  This is used only
                for controlled historical comparisons.

        Returns:
            Trained model object.
        """
        cutoff_str = pd.to_datetime(cutoff_date).strftime("%Y-%m-%d")
        if training_window_years is None and not use_config_default:
            training_window_years = None
        else:
            training_window_years = self.resolve_training_window_years(
                training_window_years
            )
        logger.info(f"Training model with cutoff={cutoff_str} (train < cutoff)")
        logger.info(
            "Layer 1 training window: %s years",
            training_window_years or "expanding",
        )

        weevil_cut = self.data_service.pull_weevil_data(today=cutoff_str)

        if training_window_years is None:
            x_train, y_train = self.data_service.prepare_training_data(
                weevil_data=weevil_cut,
                weather_data=processed_weather_unique,
                test_date=cutoff_str,
                features=self.config.features,
                target=self.config.target,
                post_event_training_days=int(
                    self.config.config.get(
                        "post_event_training_days",
                        self.config.config.get("post_event_zero_days", 5),
                    )
                ),
            )
        else:
            if not isinstance(training_window_years, int) or training_window_years <= 0:
                raise ValueError(
                    f"training_window_years must be a positive int or None, "
                    f"got {training_window_years!r}"
                )
            x_train, y_train = self.data_service.prepare_training_data(
                weevil_data=weevil_cut,
                weather_data=processed_weather_unique,
                test_date=cutoff_str,
                features=self.config.features,
                target=self.config.target,
                window_years=training_window_years,
                post_event_training_days=int(
                    self.config.config.get(
                        "post_event_training_days",
                        self.config.config.get("post_event_zero_days", 5),
                    )
                ),
            )

        model = self.model_manager.train(x_train=x_train, y_train=y_train)
        # Learned termination is intentionally trained later from earlier
        # out-of-fold trajectory predictions, not these in-sample training
        # rows.  See ``_apply_oof_termination_to_backtest_detail``.
        return model

    def _train_optional_termination_model_for_cutoff(
        self,
        cutoff_str: str,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
    ) -> None:
        settings = LearnedTerminationSettings.from_config(self.config.config)
        if not settings.enabled:
            return

        x_train, y_train = select_aligned_training_window(
            features=x_train,
            target=y_train,
            cutoff_date=cutoff_str,
            window_years=settings.training_window_years,
        )
        logger.info(
            "Training learned termination model for cutoff=%s alpha=%s threshold=%s "
            "window_years=%s rows=%s date_range=%s..%s",
            cutoff_str,
            settings.alpha,
            settings.threshold,
            settings.training_window_years or "inherited",
            len(x_train),
            pd.to_datetime(x_train.index).min().date(),
            pd.to_datetime(x_train.index).max().date(),
        )
        base_pred = self.model_manager.predict(x_train[self.config.features])
        train_frame = x_train.copy()
        train_frame["prediction_date"] = pd.to_datetime(train_frame.index)
        train_frame["raw_signed_days"] = np.asarray(base_pred, dtype=float)
        train_frame["predicted_days"] = np.clip(
            train_frame["raw_signed_days"], a_min=0.0, a_max=None
        )
        target = y_train[self.config.target].to_numpy(dtype=float)
        self._termination_models_by_cutoff[
            cutoff_str
        ] = train_learned_termination_model(
            training_frame=train_frame,
            target_days_remaining=target,
            alpha=settings.alpha,
        )

    def _load_configured_termination_model(self, settings: LearnedTerminationSettings):
        if not settings.model_path:
            return None
        try:
            return self.s3_manager.read_file(settings.model_path)
        except Exception as exc:
            logger.warning("Could not load configured termination model: %s", exc)
            return None

    def _apply_optional_termination_to_predictions(
        self,
        predictions: pd.DataFrame,
    ) -> pd.DataFrame:
        settings = LearnedTerminationSettings.from_config(self.config.config)
        if not settings.enabled or predictions is None or predictions.empty:
            return predictions

        pred = predictions.copy()
        if "model_train_cutoff_date" in pred.columns:
            parts = []
            fallback_model = None
            for cutoff, group in pred.groupby("model_train_cutoff_date", sort=False):
                cutoff_str = pd.to_datetime(cutoff).strftime("%Y-%m-%d")
                model = self._termination_models_by_cutoff.get(cutoff_str)
                if model is None:
                    if fallback_model is None:
                        fallback_model = self._load_configured_termination_model(
                            settings
                        )
                    model = fallback_model
                if model is None:
                    parts.append(group)
                    continue
                parts.append(
                    apply_learned_termination_rule(
                        predictions=group,
                        model=model,
                        threshold=settings.threshold,
                        overwrite_business_prediction=settings.overwrite_business_prediction,
                        min_history_days=settings.min_history_days,
                    )
                )
            return pd.concat(parts, ignore_index=True) if parts else pred

        model = self._load_configured_termination_model(settings)
        if model is None:
            return pred
        return apply_learned_termination_rule(
            predictions=pred,
            model=model,
            threshold=settings.threshold,
            overwrite_business_prediction=settings.overwrite_business_prediction,
            min_history_days=settings.min_history_days,
        )

    def _apply_oof_termination_to_backtest_detail(
        self, detail: pd.DataFrame, weevil_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calibrate each validation year with only prior OOF trajectories."""
        settings = LearnedTerminationSettings.from_config(self.config.config)
        if not settings.enabled or detail.empty:
            return detail
        out = detail.copy()
        out["prediction_date"] = pd.to_datetime(out["prediction_date"])
        out["location_id"] = out["location_id"].astype(str)
        out["stage_id"] = pd.to_numeric(out["stage_id"], errors="coerce").astype(int)
        events = weevil_data.copy()
        events["stage_date"] = pd.to_datetime(events["stage_date"], errors="coerce")
        events["location_id"] = events["location_id"].astype(str)
        events["stage_id"] = pd.to_numeric(events["stage_id"], errors="coerce").astype(
            int
        )
        events["test_year"] = events["stage_date"].dt.year
        actual = (
            events.sort_values("stage_date")
            .drop_duplicates(["test_year", "location_id", "stage_id"], keep="first")[
                ["test_year", "location_id", "stage_id", "stage_date"]
            ]
            .rename(columns={"stage_date": "actual_stage_date"})
        )
        # Some callers (the unified cache generator) have already joined the
        # actual date for trajectory diagnostics.  Replace it deliberately so
        # pandas cannot create ``actual_stage_date_x/y`` and make the OOF
        # target disappear below.
        out = out.drop(columns=["actual_stage_date"], errors="ignore")
        out = out.merge(actual, on=["test_year", "location_id", "stage_id"], how="left")
        out["actual_days_to_event"] = (
            pd.to_datetime(out["actual_stage_date"]) - out["prediction_date"]
        ).dt.total_seconds() / 86400.0
        parts = []
        for year in sorted(out["test_year"].dropna().astype(int).unique()):
            current = out[out["test_year"].eq(year)].copy()
            history = out[out["test_year"].lt(year)].copy()
            history = history[history["actual_days_to_event"].notna()]
            if history.empty:
                # No prior OOF trajectories: retain base output rather than
                # silently fitting Ridge on same-season/in-sample rows.
                parts.append(current)
                continue
            model = train_learned_termination_model(
                training_frame=history,
                target_days_remaining=history["actual_days_to_event"].to_numpy(float),
                alpha=settings.alpha,
            )
            parts.append(
                apply_learned_termination_rule(
                    current,
                    model,
                    settings.threshold,
                    settings.overwrite_business_prediction,
                    settings.min_history_days,
                )
            )
        return pd.concat(parts, ignore_index=True) if parts else out

    def predict_with_model_for_date(
        self,
        model,
        test_date,
        locations_for_date: List[str],
        model_train_cutoff_date: str,
        processed_weather_unique: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Predict on test_date using a pre-trained model.

        Output columns include prediction_year, model_train_cutoff_date, and
        model_train_end_year for traceability in evaluation.

        Args:
            model: Trained model object.
            test_date: Date-like; the prediction date.
            locations_for_date: Location IDs to predict for on this date.
            model_train_cutoff_date: ISO date string of the training cutoff.
            processed_weather_unique: Pre-processed, deduplicated weather DataFrame.

        Returns:
            DataFrame with prediction columns, or empty DataFrame if no predictions.
        """
        test_date_str = pd.to_datetime(test_date).strftime("%Y-%m-%d")

        # Build test set for the day
        x_test_full = self.data_service.prepare_test_data(
            weather_data=processed_weather_unique,
            features=self.config.features,
            start_date=test_date_str,
            end_date=test_date_str,
        )

        if x_test_full is None or x_test_full.empty:
            logger.warning(
                f"{test_date_str}: prepare_test_data returned empty "
                f"(no weather coverage)."
            )
            return pd.DataFrame()

        if "location_id" not in x_test_full.columns:
            logger.warning(f"{test_date_str}: x_test_full missing location_id column.")
            return pd.DataFrame()

        # Filter to scheduled locations
        loc_set = set(map(str, locations_for_date))
        x_test_full = x_test_full[
            x_test_full["location_id"].astype(str).isin(loc_set)
        ].copy()

        if x_test_full.empty:
            logger.warning(
                f"{test_date_str}: no rows after filtering to "
                f"{len(loc_set)} locations."
            )
            return pd.DataFrame()

        # Predict
        x_test_model = x_test_full[self.config.features].copy()
        self.model_manager.model = model
        y_pred = self.model_manager.predict(x_test_model)
        raw_signed_days = np.asarray(y_pred, dtype=float)
        predicted_days = np.clip(raw_signed_days, a_min=0.0, a_max=None)

        # Build output frame
        out = x_test_full[["location_id", "stage_id", "latitude", "longitude"]].copy()
        out["prediction_date"] = test_date_str
        out["prediction_year"] = int(pd.to_datetime(test_date_str).year)
        out["model_train_cutoff_date"] = model_train_cutoff_date
        out["model_train_end_year"] = (
            int(pd.to_datetime(model_train_cutoff_date).year) - 1
        )
        out["raw_signed_days"] = raw_signed_days
        out["predicted_days"] = predicted_days

        pred_dates = pd.to_datetime(test_date_str) + pd.to_timedelta(
            predicted_days, unit="D"
        )
        out["predicted_stage_date"] = (
            pd.to_datetime(pred_dates).astype("datetime64[ns]").astype(str).str[:10]
        )

        return out

    # ------------------------------------------------------------------
    # Step 3b — Legacy per-date prediction (expanding/rolling, train each date)
    # ------------------------------------------------------------------

    def predict_for_date(
        self,
        test_date,
        locations_for_date: List[str],
        processed_weather_unique: pd.DataFrame,
        training_window_years: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Train up to *test_date* and predict on *test_date* for selected locations.
        Trains a fresh model for every call (slower than production mode).

        Args:
            test_date: The testing date (date-like)
            locations_for_date: Location IDs to predict for on this date
            processed_weather_unique: Pre-processed, deduplicated weather
            training_window_years:
                None -> use the config default, or expanding when absent
                int  -> fixed rolling window (last N years)

        Returns:
            DataFrame with prediction columns, or empty DataFrame if no predictions.
        """
        test_date_str = pd.to_datetime(test_date).strftime("%Y-%m-%d")
        training_window_years = self.resolve_training_window_years(
            training_window_years
        )

        weevil_cut = self.data_service.pull_weevil_data(today=test_date_str)

        if training_window_years is None:
            x_train, y_train = self.data_service.prepare_training_data(
                weevil_data=weevil_cut,
                weather_data=processed_weather_unique,
                test_date=test_date_str,
                features=self.config.features,
                target=self.config.target,
                post_event_training_days=int(
                    self.config.config.get(
                        "post_event_training_days",
                        self.config.config.get("post_event_zero_days", 5),
                    )
                ),
            )
        else:
            x_train, y_train = self.data_service.prepare_training_data(
                weevil_data=weevil_cut,
                weather_data=processed_weather_unique,
                test_date=test_date_str,
                features=self.config.features,
                target=self.config.target,
                window_years=training_window_years,
                post_event_training_days=int(
                    self.config.config.get(
                        "post_event_training_days",
                        self.config.config.get("post_event_zero_days", 5),
                    )
                ),
            )

        model = self.model_manager.train(x_train=x_train, y_train=y_train)

        x_test_full = self.data_service.prepare_test_data(
            weather_data=processed_weather_unique,
            features=self.config.features,
            start_date=test_date_str,
            end_date=test_date_str,
        )

        if x_test_full is None or x_test_full.empty:
            logger.warning(f"{test_date_str}: prepare_test_data returned empty.")
            return pd.DataFrame()

        if "location_id" not in x_test_full.columns:
            logger.warning(f"{test_date_str}: x_test_full has no location_id column.")
            return pd.DataFrame()

        loc_set = set(map(str, locations_for_date))
        x_test_full = x_test_full[
            x_test_full["location_id"].astype(str).isin(loc_set)
        ].copy()

        if x_test_full.empty:
            logger.warning(
                f"{test_date_str}: no rows after filtering to "
                f"{len(loc_set)} locations."
            )
            return pd.DataFrame()

        x_test_model = x_test_full[self.config.features].copy()
        y_pred = self.model_manager.predict(x_test_model)
        raw_signed_days = np.asarray(y_pred, dtype=float)
        predicted_days = np.clip(raw_signed_days, a_min=0.0, a_max=None)

        out = x_test_full[["location_id", "stage_id", "latitude", "longitude"]].copy()
        out["prediction_date"] = test_date_str
        out["raw_signed_days"] = raw_signed_days
        out["predicted_days"] = predicted_days

        pred_dates = pd.to_datetime(test_date_str) + pd.to_timedelta(
            predicted_days, unit="D"
        )
        out["predicted_stage_date"] = (
            pd.to_datetime(pred_dates).astype("datetime64[ns]").astype(str).str[:10]
        )

        return out

    # ------------------------------------------------------------------
    # Main pipeline: plan -> weather -> production loop -> save
    # ------------------------------------------------------------------

    def run_backtest(
        self,
        years: List[int] = [2023, 2024, 2025],
        stages: List[int] = [1, 4, 2, 3],
        lead_days: int = 7,
        training_window_years: Optional[int] = None,
        upload_to_s3: bool = True,
        delete_local: bool = True,
    ) -> str:
        """
        Full production backtesting pipeline: plan -> weather -> loop -> save.

        Production mode: train ONE model per prediction year (expanding window,
        cutoff = Jan 1 of that year) and reuse it for all testing_dates in the year.

        Args:
            years: Years to backtest
            stages: Stage IDs to include
            lead_days: Days before stage_date used as testing_date
            training_window_years:
                None -> expanding window (default, from 2018)
                int  -> fixed rolling window (last N years)
            upload_to_s3: Upload final Parquet to S3 after run
            delete_local: Delete local predictions file after S3 upload

        Returns:
            S3 path (if uploaded) or local Parquet file path
        """
        logger.info(f"Starting production backtesting for years={years}")

        # -- Step 1: Build plan --
        date_to_locations, events, weevil_data = self.build_backtesting_plan(
            years=years, stages=stages, lead_days=lead_days
        )

        test_dates = sorted(date_to_locations.keys())
        if not test_dates:
            logger.warning("No test dates found. Aborting.")
            return None

        # -- Step 2: Weather (pull once, process once) --
        processed_weather_unique = self.prepare_weather(weevil_data, test_dates)

        # -- Step 3: Output file setup --
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Incremental checkpoint (CSV)
        pred_tmp_csv = (
            self.output_dir / f"predictions_detail_lead{lead_days}_{timestamp}.tmp.csv"
        )
        # Final deliverables
        pred_detail_parquet = (
            self.output_dir / f"predictions_detail_lead{lead_days}_{timestamp}.parquet"
        )
        pred_detail_parquet_latest = self.output_dir / "predictions_detail.parquet"
        summary_year_stage_csv = (
            self.output_dir
            / f"testing_results_year_stage_lead{lead_days}_{timestamp}.csv"
        )
        summary_year_stage_latest = self.output_dir / "testing_results_year_stage.csv"
        summary_year_overall_csv = (
            self.output_dir
            / f"testing_results_year_overall_lead{lead_days}_{timestamp}.csv"
        )
        summary_year_overall_latest = (
            self.output_dir / "testing_results_year_overall.csv"
        )
        run_log_path = self.output_dir / "run_log.csv"
        fail_log_path = self.output_dir / "failures.csv"

        header_written = pred_tmp_csv.exists() and pred_tmp_csv.stat().st_size > 0

        # -- Resume: skip dates already in checkpoint --
        completed_dates: Set = set()
        if pred_tmp_csv.exists() and pred_tmp_csv.stat().st_size > 0:
            try:
                tmp = pd.read_csv(pred_tmp_csv, usecols=["prediction_date"])
                completed_dates = set(
                    pd.to_datetime(tmp["prediction_date"]).dt.date.unique()
                )
                logger.info(
                    f"[Resume] Found {len(completed_dates)} completed dates "
                    f"in {pred_tmp_csv}"
                )
            except Exception as e:
                logger.warning(f"[Resume] Could not read checkpoint: {e}")

        # -- Build year -> test_dates mapping --
        test_dates_by_year: Dict[int, List] = {}
        for d in test_dates:
            d_dt = pd.to_datetime(d).date()
            test_dates_by_year.setdefault(d_dt.year, []).append(d_dt)
        for y in test_dates_by_year:
            test_dates_by_year[y] = sorted(test_dates_by_year[y])

        # -- Main production loop: train once per year --
        run_records: List[Dict] = []
        failure_records: List[Dict] = []
        total_rows = 0
        overall_t0 = datetime.now()

        for y in sorted(test_dates_by_year.keys()):
            cutoff_date = datetime(y, 1, 1)
            cutoff_str = cutoff_date.strftime("%Y-%m-%d")

            logger.info(
                f"\n==== [Year {y}] Training with cutoff={cutoff_str} "
                f"(train uses dates < cutoff) ===="
            )

            model = None
            try:
                model = self.train_model_for_cutoff(
                    cutoff_date=cutoff_date,
                    processed_weather_unique=processed_weather_unique,
                    training_window_years=training_window_years,
                )
                logger.info(f"[Year {y}] Model trained successfully")
            except Exception as e:
                logger.error(f"[Year {y}] Training failed: {e}")
                failure_records.append({"testing_date": f"{y}-TRAIN", "error": str(e)})
                pd.DataFrame(failure_records).to_csv(fail_log_path, index=False)
                continue

            # Predict for each testing_date in this year
            dates_in_year = test_dates_by_year[y]
            for i, d in enumerate(dates_in_year, start=1):
                if d in completed_dates:
                    continue

                d_str = pd.to_datetime(d).strftime("%Y-%m-%d")
                locations_for_date = date_to_locations.get(d, [])

                logger.info(
                    f"[Year {y}] [{i}/{len(dates_in_year)}] "
                    f"Backtesting date {d_str}: locations={len(locations_for_date)}"
                )

                status = "ok"
                reason = None
                aggregated = False
                n_rows = 0
                n_rows_raw = 0

                try:
                    if len(locations_for_date) == 0:
                        status = "skip"
                        reason = "no_locations_for_date"
                        logger.warning(f"{d_str}: no locations; skipping.")

                    else:
                        out_df = self.predict_with_model_for_date(
                            model=model,
                            test_date=d,
                            locations_for_date=locations_for_date,
                            model_train_cutoff_date=cutoff_str,
                            processed_weather_unique=processed_weather_unique,
                        )

                        if out_df is None or out_df.empty:
                            status = "skip"
                            reason = "no_predictions_produced"
                            logger.warning(
                                f"{d_str}: no predictions produced; skipping."
                            )

                        else:
                            n_rows_raw = len(out_df)

                            # Sanity check: expect one row per configured direct target.
                            expected = len(stages) * len(locations_for_date)
                            if n_rows_raw != expected:
                                logger.warning(
                                    f"{d_str}: rows={n_rows_raw} != expected "
                                    f"{len(stages)}x{len(locations_for_date)}={expected}"
                                )

                            # Aggregate duplicates if any
                            key_cols = ["prediction_date", "location_id", "stage_id"]
                            if out_df.duplicated(subset=key_cols, keep=False).any():
                                aggregated = True
                                out_df = self.aggregate_predictions_unique(out_df)

                            n_rows = len(out_df)

                            # Append to incremental CSV checkpoint
                            out_df.to_csv(
                                pred_tmp_csv,
                                mode="a",
                                header=not header_written,
                                index=False,
                            )
                            header_written = True
                            total_rows += n_rows
                            completed_dates.add(d)

                            if aggregated:
                                logger.info(
                                    f"{d_str}: saved {n_rows} unique rows "
                                    f"(raw={n_rows_raw}, "
                                    f"dup_factor={n_rows_raw / max(n_rows, 1):.2f})"
                                )
                            else:
                                logger.info(
                                    f"{d_str}: saved {n_rows} rows "
                                    f"(no aggregation needed)"
                                )

                except Exception as e:
                    status = "fail"
                    reason = str(e)
                    logger.error(f"{d_str}: Failed: {e}")
                    failure_records.append({"testing_date": d_str, "error": str(e)})

                run_records.append(
                    {
                        "prediction_year": y,
                        "model_train_cutoff_date": cutoff_str,
                        "testing_date": d_str,
                        "n_locations": len(locations_for_date),
                        "n_predictions": n_rows,
                        "n_predictions_raw": n_rows_raw,
                        "aggregated": aggregated,
                        "status": status,
                        "reason": reason,
                    }
                )

                # Checkpoint logs every date (safe for nohup / interruptions)
                pd.DataFrame(run_records).to_csv(run_log_path, index=False)
                pd.DataFrame(failure_records).to_csv(fail_log_path, index=False)

        # -- Final summary --
        overall_dt = (datetime.now() - overall_t0).total_seconds()
        logger.info(f"Backtesting completed. Total predictions: {total_rows}")
        logger.info(f"Total runtime: {overall_dt:.1f}s")

        run_df = pd.DataFrame(run_records)
        run_df.to_csv(self.output_dir / "run_log_final.csv", index=False)

        summary = (
            run_df.groupby("status")["testing_date"]
            .count()
            .rename("n_dates")
            .reset_index()
        )
        summary.to_csv(self.output_dir / "run_summary.csv", index=False)
        logger.info(f"Run summary:\n{summary.to_string(index=False)}")

        if len(failure_records) > 0:
            pd.DataFrame(failure_records).to_csv(
                self.output_dir / "failures_final.csv", index=False
            )

        # -- Convert incremental CSV to Parquet --
        result_path = str(pred_tmp_csv)
        if pred_tmp_csv.exists() and pred_tmp_csv.stat().st_size > 0:
            try:
                pred_detail = pd.read_csv(pred_tmp_csv)
                pred_detail["prediction_date"] = pd.to_datetime(
                    pred_detail["prediction_date"], errors="coerce"
                )
                pred_detail["predicted_stage_date"] = pd.to_datetime(
                    pred_detail["predicted_stage_date"], errors="coerce"
                )
                pred_detail["location_id"] = pred_detail["location_id"].astype(str)
                pred_detail["stage_id"] = pd.to_numeric(
                    pred_detail["stage_id"], errors="coerce"
                ).astype(int)
                pred_detail = self._apply_oof_termination_to_backtest_detail(
                    pred_detail, weevil_data
                )
                pred_detail["predicted_stage_date"] = pd.to_datetime(
                    pred_detail["predicted_stage_date"], errors="coerce"
                )

                pred_detail.to_parquet(pred_detail_parquet, index=False)
                pred_detail.to_parquet(pred_detail_parquet_latest, index=False)
                logger.info(f"Saved Parquet: {pred_detail_parquet}")
                result_path = str(pred_detail_parquet)
            except Exception as e:
                logger.warning(f"to_parquet failed ({e}); keeping CSV: {pred_tmp_csv}")

        # -- Optional S3 upload --
        s3_path = None
        if upload_to_s3:
            s3_path = self._upload_to_s3(result_path, timestamp, lead_days)
            if delete_local and s3_path:
                self._delete_local_file(result_path)

        return s3_path if s3_path else result_path

    @staticmethod
    def summarize_event_window_validation(
        detail: pd.DataFrame,
        termination_threshold: float = 0.0,
        near_zero_threshold: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Summarize event-window validation results and list non-terminated windows.
        """
        if detail is None or detail.empty:
            return pd.DataFrame(), pd.DataFrame()

        df = detail.copy()
        df["predicted_days"] = pd.to_numeric(df["predicted_days"], errors="coerce")
        df["abs_error_days"] = pd.to_numeric(df["abs_error_days"], errors="coerce")
        df["terminated"] = df["predicted_days"] <= termination_threshold

        if near_zero_threshold is None:
            near_zero_threshold = termination_threshold
        df["near_zero"] = df["predicted_days"] <= near_zero_threshold

        group_cols = ["event_year", "stage_id", "location_id", "stage_date"]
        window_summary = df.groupby(group_cols, as_index=False).agg(
            n_prediction_rows=("predicted_days", "size"),
            min_predicted_days=("predicted_days", "min"),
            day0_predicted_days=(
                "predicted_days",
                lambda s: (
                    s[df.loc[s.index, "days_from_event"] == 0].iloc[0]
                    if (df.loc[s.index, "days_from_event"] == 0).any()
                    else np.nan
                ),
            ),
            median_abs_error_days=("abs_error_days", "median"),
            mean_abs_error_days=("abs_error_days", "mean"),
            p90_abs_error_days=(
                "abs_error_days",
                lambda s: (
                    float(np.percentile(s.dropna().astype(float), 90))
                    if len(s.dropna())
                    else np.nan
                ),
            ),
            terminated_within_window=("terminated", "max"),
            near_zero_within_window=("near_zero", "max"),
        )

        summary_year_stage = window_summary.groupby(
            ["event_year", "stage_id"], as_index=False
        ).agg(
            n_event_windows=("location_id", "size"),
            n_prediction_rows=("n_prediction_rows", "sum"),
            terminated_windows=("terminated_within_window", "sum"),
            near_zero_windows=("near_zero_within_window", "sum"),
            median_min_predicted_days=("min_predicted_days", "median"),
            median_day0_predicted_days=("day0_predicted_days", "median"),
            median_abs_error_days=("median_abs_error_days", "median"),
            mean_abs_error_days=("mean_abs_error_days", "mean"),
            p90_abs_error_days=("p90_abs_error_days", "median"),
        )
        summary_year_stage["termination_rate"] = summary_year_stage[
            "terminated_windows"
        ] / summary_year_stage["n_event_windows"].replace(0, np.nan)
        summary_year_stage["near_zero_rate"] = summary_year_stage[
            "near_zero_windows"
        ] / summary_year_stage["n_event_windows"].replace(0, np.nan)
        summary_year_stage["termination_threshold"] = termination_threshold
        summary_year_stage["near_zero_threshold"] = near_zero_threshold
        summary_year_stage = summary_year_stage.sort_values(
            ["event_year", "stage_id"]
        ).reset_index(drop=True)

        non_terminated = window_summary[
            ~window_summary["terminated_within_window"].fillna(False)
        ].copy()
        non_terminated = non_terminated.sort_values(
            ["event_year", "stage_id", "median_abs_error_days", "location_id"],
            ascending=[True, True, False, True],
        ).reset_index(drop=True)

        return summary_year_stage, non_terminated

    def run_event_window_validation(
        self,
        years: List[int] = [2023, 2024, 2025],
        stages: List[int] = [1, 4, 2, 3],
        days_before: int = 3,
        days_after: int = 3,
        training_window_years: Optional[int] = None,
        termination_threshold: float = 0.0,
        near_zero_threshold: Optional[float] = None,
        upload_to_s3: bool = True,
        delete_local: bool = False,
    ) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
        """
        Run event-centered validation for the same event scope as backtesting.

        Unlike ``run_backtest`` (one prediction date per event at a fixed lead),
        this method expands each selected event into a prediction window around
        the actual stage date. It is useful for signed-target validation because
        it measures whether predictions terminate near the true event boundary.
        """
        logger.info(
            "Starting event-window validation for years=%s stages=%s window=[-%s,+%s]",
            years,
            stages,
            days_before,
            days_after,
        )

        _, events, weevil_data = self.build_backtesting_plan(
            years=years,
            stages=stages,
            lead_days=days_before,
        )
        event_windows = self.build_event_window_plan(
            events=events,
            days_before=days_before,
            days_after=days_after,
        )
        if event_windows.empty:
            raise RuntimeError("No event-window rows were generated.")

        test_dates = sorted(event_windows["prediction_date_only"].unique().tolist())
        processed_weather_unique = self.prepare_weather(weevil_data, test_dates)

        detail_parts: List[pd.DataFrame] = []
        for event_year in sorted(event_windows["event_year"].unique()):
            cutoff_date = datetime(int(event_year), 1, 1)
            cutoff_str = cutoff_date.strftime("%Y-%m-%d")
            logger.info(
                "[Event year %s] Training with cutoff=%s", event_year, cutoff_str
            )
            model = self.train_model_for_cutoff(
                cutoff_date=cutoff_date,
                processed_weather_unique=processed_weather_unique,
                training_window_years=training_window_years,
            )

            year_windows = event_windows[
                event_windows["event_year"] == event_year
            ].copy()
            prediction_dates = sorted(
                year_windows["prediction_date_only"].unique().tolist()
            )
            for d in prediction_dates:
                date_mask = year_windows["prediction_date_only"] == d
                locations_for_date = sorted(
                    year_windows.loc[date_mask, "location_id"]
                    .astype(str)
                    .unique()
                    .tolist()
                )
                out_df = self.predict_with_model_for_date(
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
                    out_df = self.aggregate_predictions_unique(out_df)

                out_df = out_df.copy()
                out_df["prediction_date"] = pd.to_datetime(
                    out_df["prediction_date"]
                ).dt.normalize()
                out_df["location_id"] = out_df["location_id"].astype(str)
                out_df["stage_id"] = pd.to_numeric(
                    out_df["stage_id"], errors="coerce"
                ).astype(int)

                detail_parts.append(
                    year_windows[date_mask].merge(
                        out_df[
                            [
                                "location_id",
                                "stage_id",
                                "prediction_date",
                                "prediction_year",
                                "model_train_cutoff_date",
                                "model_train_end_year",
                                "raw_signed_days",
                                "predicted_days",
                                "predicted_stage_date",
                            ]
                        ],
                        on=["location_id", "stage_id", "prediction_date"],
                        how="left",
                    )
                )

        if not detail_parts:
            raise RuntimeError(
                "No predictions were produced for event-window validation."
            )

        detail = pd.concat(detail_parts, ignore_index=True)
        detail = self._apply_optional_termination_to_predictions(detail)
        detail["actual_stage_date"] = pd.to_datetime(
            detail["stage_date"], errors="coerce"
        ).dt.normalize()
        detail["predicted_stage_date"] = pd.to_datetime(
            detail["predicted_stage_date"], errors="coerce"
        ).dt.normalize()
        detail["error_days"] = (
            detail["predicted_stage_date"] - detail["actual_stage_date"]
        ).dt.days
        detail["abs_error_days"] = pd.to_numeric(
            detail["error_days"], errors="coerce"
        ).abs()
        detail["predicted_days"] = pd.to_numeric(
            detail["predicted_days"], errors="coerce"
        )
        detail["terminated"] = detail["predicted_days"] <= termination_threshold
        if near_zero_threshold is None:
            near_zero_threshold = float(
                self.config.config.get(
                    "zero_day_termination_threshold",
                    termination_threshold,
                )
            )
        detail["near_zero"] = detail["predicted_days"] <= near_zero_threshold
        detail = detail.sort_values(
            ["event_year", "location_id", "stage_id", "stage_date", "prediction_date"]
        ).reset_index(drop=True)

        summary_year_stage, non_terminated = self.summarize_event_window_validation(
            detail=detail,
            termination_threshold=termination_threshold,
            near_zero_threshold=near_zero_threshold,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stage_label = "_".join(map(str, stages))
        window_label = f"days_before{days_before}_after{days_after}"
        detail_path = (
            self.output_dir
            / f"event_window_predictions_stages{stage_label}_{window_label}_{timestamp}.parquet"
        )
        detail_latest = self.output_dir / "event_window_predictions.parquet"
        summary_path = (
            self.output_dir
            / f"event_window_summary_year_stage_{window_label}_{timestamp}.csv"
        )
        summary_latest = self.output_dir / "event_window_summary_year_stage.csv"
        non_terminated_path = (
            self.output_dir
            / f"event_window_non_terminated_windows_{window_label}_{timestamp}.csv"
        )
        non_terminated_latest = (
            self.output_dir / "event_window_non_terminated_windows.csv"
        )

        detail.to_parquet(detail_path, index=False)
        detail.to_parquet(detail_latest, index=False)
        summary_year_stage.to_csv(summary_path, index=False)
        summary_year_stage.to_csv(summary_latest, index=False)
        non_terminated.to_csv(non_terminated_path, index=False)
        non_terminated.to_csv(non_terminated_latest, index=False)

        logger.info("Event-window detail saved: %s", detail_path)
        logger.info("Event-window summary saved: %s", summary_path)
        logger.info(
            "Event-window non-terminated windows saved: %s", non_terminated_path
        )

        if upload_to_s3:
            output_path = self.config.config.get("output_path", "")
            for local_path in [detail_path, summary_path, non_terminated_path]:
                if local_path.suffix == ".parquet":
                    data = pd.read_parquet(local_path)
                else:
                    data = pd.read_csv(local_path)
                s3_key = f"{output_path}/{local_path.name}"
                self.s3_manager.upload_file(data, s3_key)
                logger.info("Uploaded to S3: %s", s3_key)
                if delete_local:
                    self._delete_local_file(str(local_path))

        return str(detail_path), summary_year_stage, non_terminated

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_7day_forecast_error(
        self,
        predictions_file: Optional[str] = None,
        years: List[int] = [2023, 2024, 2025],
        stages: List[int] = [1, 4, 2, 3],
        lead_days: int = 7,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Evaluate forecasting error.

        Matching rule:
          expected_actual_stage_date = prediction_date + lead_days
          merge on (location_id, stage_id, expected == actual stage_date)
          error_days = predicted_stage_date - actual_stage_date

        Args:
            predictions_file: Path to backtest predictions (Parquet or CSV)
            years: Years to filter actual stage data
            stages: Stage IDs to evaluate
            lead_days: Lead days used in backtesting

        Returns:
            (evaluation_df, summary_year_stage_df, summary_year_overall_df)
        """
        logger.info("Evaluating forecasting error...")

        if predictions_file is None:
            raise ValueError("predictions_file must be provided")

        # Handle S3 paths
        if predictions_file.startswith("s3://") or (
            not os.path.exists(predictions_file) and "/" in predictions_file
        ):
            logger.info(f"Downloading predictions from S3: {predictions_file}")
            local_temp = (
                self.output_dir
                / f"temp_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            )
            try:
                pred_downloaded = self.s3_manager.read_file(predictions_file)
                pred_downloaded.to_parquet(local_temp, index=False)
                predictions_file = str(local_temp)
            except Exception as e:
                logger.error(f"Error downloading from S3: {e}")
                raise

        if not os.path.exists(predictions_file):
            raise FileNotFoundError(f"Predictions file not found: {predictions_file}")

        # Load predictions: prefer Parquet, fallback to CSV
        if predictions_file.endswith(".parquet"):
            pred_df = pd.read_parquet(predictions_file)
        else:
            pred_df = pd.read_csv(predictions_file)

        pred_df["prediction_date"] = pd.to_datetime(pred_df["prediction_date"])
        pred_df["predicted_stage_date"] = pd.to_datetime(
            pred_df["predicted_stage_date"]
        )
        pred_df["location_id"] = pred_df["location_id"].astype(str)
        pred_df["stage_id"] = pd.to_numeric(
            pred_df["stage_id"], errors="coerce"
        ).astype(int)

        # Ensure prediction_year column exists
        if "prediction_year" not in pred_df.columns:
            pred_df["prediction_year"] = pred_df["prediction_date"].dt.year

        logger.info(f"Loaded {len(pred_df)} predictions")

        # Pull actual stage dates (with location metadata)
        max_pred_date = pred_df["prediction_date"].max() + timedelta(days=lead_days + 2)
        weevil_eval = self.data_service.pull_weevil_data(
            today=max_pred_date.strftime("%Y-%m-%d")
        )
        weevil_eval["stage_date"] = pd.to_datetime(weevil_eval["stage_date"])
        weevil_eval["location_id"] = weevil_eval["location_id"].astype(str)
        weevil_eval["stage_id"] = pd.to_numeric(
            weevil_eval["stage_id"], errors="coerce"
        ).astype(int)

        # Select columns, including optional location metadata
        base_cols = [
            "location_id",
            "stage_id",
            "stage_date",
            "year",
            "latitude",
            "longitude",
        ]
        optional_cols = ["location_state", "location_name"]
        actual_cols = base_cols + [c for c in optional_cols if c in weevil_eval.columns]

        actual = weevil_eval[
            (weevil_eval["year"].isin(years)) & (weevil_eval["stage_id"].isin(stages))
        ][actual_cols].copy()

        actual["stage_date_only"] = actual["stage_date"].dt.date
        actual = (
            actual.sort_values(["location_id", "stage_id", "stage_date"])
            .drop_duplicates(
                ["location_id", "stage_id", "stage_date_only"], keep="first"
            )
            .copy()
        )

        # Merge: prediction_date + lead_days == actual stage_date
        pred_df["expected_actual_stage_date"] = (
            pred_df["prediction_date"] + timedelta(days=lead_days)
        ).dt.date

        merged = pred_df.merge(
            actual,
            left_on=["location_id", "stage_id", "expected_actual_stage_date"],
            right_on=["location_id", "stage_id", "stage_date_only"],
            how="inner",
        )

        logger.info(
            f"Matched {len(merged)} predictions to actual stage dates "
            f"(out of {len(pred_df)} total predictions)"
        )

        if merged.empty:
            logger.warning(
                "No predictions matched. " "Check location_id and date alignment."
            )
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Calculate error
        merged["error_days"] = (
            merged["predicted_stage_date"] - merged["stage_date"]
        ).dt.days

        # Build evaluation detail
        detail_cols = [
            "location_id",
            "stage_id",
            "prediction_year",
        ]
        for col in ["model_train_end_year", "model_train_cutoff_date"]:
            if col in merged.columns:
                detail_cols.append(col)
        detail_cols += [
            "prediction_date",
            "predicted_stage_date",
            "stage_date",
            "error_days",
        ]
        lat_col = "latitude_x" if "latitude_x" in merged.columns else "latitude"
        lon_col = "longitude_x" if "longitude_x" in merged.columns else "longitude"
        detail_cols += [lat_col, lon_col]
        for col in ["location_state", "location_name"]:
            if col in merged.columns:
                detail_cols.append(col)

        evaluation_df = merged[detail_cols].copy()
        evaluation_df.rename(
            columns={
                "stage_date": "actual_stage_date",
                "latitude_x": "latitude",
                "longitude_x": "longitude",
            },
            inplace=True,
        )

        # -- Summary: year × stage --
        def _p90_abs(x: pd.Series) -> float:
            x = x.dropna().astype(float)
            return float(np.percentile(np.abs(x), 90)) if len(x) else np.nan

        gb_ys = evaluation_df.groupby(["prediction_year", "stage_id"], dropna=False)
        summary_year_stage_df = (
            gb_ys["error_days"]
            .apply(
                lambda s: pd.Series(
                    {
                        "n_predictions": int(len(s)),
                        "mean_error": float(s.mean()),
                        "median_error": float(s.median()),
                        "mae_error": float(np.abs(s).mean()),
                        "within_3d": float((np.abs(s) <= 3).mean()),
                        "within_5d": float((np.abs(s) <= 5).mean()),
                        "p90_abs_error": _p90_abs(s),
                    }
                )
            )
            .reset_index()
            .sort_values(["prediction_year", "stage_id"])
        )

        # -- Summary: year overall --
        gb_y = evaluation_df.groupby(["prediction_year"], dropna=False)
        summary_year_overall_df = (
            gb_y["error_days"]
            .apply(
                lambda s: pd.Series(
                    {
                        "n_predictions": int(len(s)),
                        "mean_error": float(s.mean()),
                        "median_error": float(s.median()),
                        "mae_error": float(np.abs(s).mean()),
                        "within_3d": float((np.abs(s) <= 3).mean()),
                        "within_5d": float((np.abs(s) <= 5).mean()),
                        "p90_abs_error": _p90_abs(s),
                    }
                )
            )
            .reset_index()
            .sort_values(["prediction_year"])
        )

        # Save results
        self._save_evaluation_results(
            evaluation_df,
            summary_year_stage_df,
            summary_year_overall_df,
            lead_days,
        )

        print("\n" + "=" * 60)
        print(f"{lead_days}-Day Forecasting Error — Year × Stage")
        print("=" * 60)
        print(summary_year_stage_df.to_string(index=False))
        print("\n" + "=" * 60)
        print(f"{lead_days}-Day Forecasting Error — Year Overall")
        print("=" * 60)
        print(summary_year_overall_df.to_string(index=False))
        print("=" * 60 + "\n")

        return evaluation_df, summary_year_stage_df, summary_year_overall_df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_evaluation_results(
        self,
        evaluation_df: pd.DataFrame,
        summary_year_stage_df: pd.DataFrame,
        summary_year_overall_df: pd.DataFrame,
        lead_days: int = 7,
    ) -> None:
        """Save evaluation CSVs locally (timestamped + latest) & upload to S3."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        eval_file = self.output_dir / f"evaluation_lead{lead_days}_{ts}.csv"
        ys_file = (
            self.output_dir / f"testing_results_year_stage_lead{lead_days}_{ts}.csv"
        )
        yo_file = (
            self.output_dir / f"testing_results_year_overall_lead{lead_days}_{ts}.csv"
        )
        ys_latest = self.output_dir / "testing_results_year_stage.csv"
        yo_latest = self.output_dir / "testing_results_year_overall.csv"

        evaluation_df.to_csv(eval_file, index=False)
        summary_year_stage_df.to_csv(ys_file, index=False)
        summary_year_stage_df.to_csv(ys_latest, index=False)
        summary_year_overall_df.to_csv(yo_file, index=False)
        summary_year_overall_df.to_csv(yo_latest, index=False)

        logger.info(f"Evaluation detail saved: {eval_file}")
        logger.info(f"Year×stage summary saved: {ys_file}")
        logger.info(f"Year-overall summary saved: {yo_file}")

        try:
            output_path = self.config.config.get("output_path", "")
            for local, key in [
                (evaluation_df, f"evaluation_lead{lead_days}_{ts}.csv"),
                (
                    summary_year_stage_df,
                    f"testing_results_year_stage_lead{lead_days}_{ts}.csv",
                ),
                (
                    summary_year_overall_df,
                    f"testing_results_year_overall_lead{lead_days}_{ts}.csv",
                ),
            ]:
                s3_key = f"{output_path}/{key}"
                self.s3_manager.upload_file(local, s3_key)
                logger.info(f"Uploaded to S3: {s3_key}")
        except Exception as e:
            logger.warning(f"Could not upload evaluation to S3: {e}")

    def _upload_to_s3(self, local_file: str, timestamp: str, lead_days: int = 7) -> str:
        """Upload predictions file (CSV or Parquet) to S3."""
        logger.info(f"Uploading {local_file} to S3...")
        try:
            ext = Path(local_file).suffix
            if ext == ".parquet":
                data = pd.read_parquet(local_file)
            else:
                data = pd.read_csv(local_file)
            output_path = self.config.config.get("output_path", "")
            fname = Path(local_file).name
            s3_path = f"{output_path}/{fname}"
            self.s3_manager.upload_file(data, s3_path)
            logger.info(f"Uploaded to S3: {s3_path}")
            return s3_path
        except Exception as e:
            logger.error(f"S3 upload error: {e}")
            raise

    def _delete_local_file(self, local_file: str) -> None:
        """Delete a local file if it exists."""
        try:
            if os.path.exists(local_file):
                os.remove(local_file)
                logger.info(f"Deleted local file: {local_file}")
        except Exception as e:
            logger.error(f"Error deleting {local_file}: {e}")


# ======================================================================
# CLI entry point
# ======================================================================
if __name__ == "__main__":
    logger.info("Starting backtesting...")

    backtester = BacktestingFramework(
        config_path="app/config/weeviltrak_v2.4.yml",
        output_dir="outputs/backtesting",
    )

    # Run full backtest (production mode: train once per year)
    # Config default: v2.4 uses the recent three-year window with a coverage
    # gate; an explicit override remains available for experiments.
    result_path = backtester.run_backtest(
        years=[2023, 2024, 2025],
        stages=[1, 4, 2, 3],
        lead_days=7,
        training_window_years=None,
        upload_to_s3=True,
        delete_local=True,
    )

    # Evaluate
    (
        evaluation_df,
        summary_year_stage_df,
        summary_year_overall_df,
    ) = backtester.evaluate_7day_forecast_error(
        predictions_file=result_path,
        years=[2023, 2024, 2025],
    )

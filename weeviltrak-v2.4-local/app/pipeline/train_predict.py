"""
Goal: Manage end-to-end workflow for WeevilTrak model training + prediction.
Input: weevil observations (S3), weather data (S3/Redshift), spatial coverage (S3), config (YAML).
Output: trained model saved to S3; prediction CSV saved to S3; cached parquet files for restartable prediction.
Key keys: (date, location_id) for training weather; (centroid_lat, centroid_lon) for coverage->weather join; (stage_id, lat, lon) in prediction output.
Default production config is ``app/config/weeviltrak_v2.4.yml`` (LightGBM with
``signed_days_to_event`` target and canonical Stage/Phase business output).
Predictions retain both ``raw_signed_days`` and business-facing
``predicted_days = max(raw_signed_days, 0)``.
How to run: Instantiate WeevilTrakPipeline(config_path) then run_training(...), run_prediction(...).
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
import re
import gc
from pathlib import Path
import pyarrow.parquet as pq
import shutil


from app.pipeline.base import PipelineBase
from app.pipeline.termination import (
    LearnedTerminationSettings,
    apply_learned_termination_rule,
    select_aligned_training_window,
    train_learned_termination_model,
)
from app.services.canonical_events import (
    add_stage_output_names,
    canonical_events_enabled,
)
from app.models.factory import create_model_manager
from app.services.training_coverage import (
    RecentLabelCoveragePolicy,
    require_recent_label_coverage,
)
from app.settings import setup_logger

logger = setup_logger()


class WeevilTrakPipeline(PipelineBase):
    """
    Pipeline orchestrator for WeevilTrak model training and prediction.
    """

    def __init__(self, config_path: str):
        """
        Initialize pipeline with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        super().__init__(config_path)

    def run_training(
        self,
        today: Optional[str] = None,
        window_years: Optional[int] = None,
        include_today: bool = True,
        model_path_override: Optional[str] = None,
    ) -> None:
        """
        Execute training pipeline and save model to S3.

        Args:
            today: Cutoff date for training (defaults to today).
            window_years:
                - None: use ``training_window_years`` from config, or expanding
                  when the config does not define a default
                - int (e.g., 5): rolling window (last N years ending at `today`)
            include_today: Whether to include samples dated exactly ``today``.
            model_path_override: Optional S3/local path to save the trained model.
        """
        if today is None:
            today = datetime.now().strftime("%Y-%m-%d")
        today_str = pd.to_datetime(today).strftime("%Y-%m-%d")

        window_years = self.resolve_training_window_years(window_years)
        logger.info("Layer 1 training window: %s years", window_years or "expanding")

        try:
            # Step 1: Pull weevil data (used to build y_train)
            logger.info("Step 1: Pulling weevil data from S3...")
            weevil_data = self.data_service.pull_weevil_data(today=today_str)

            coverage_policy = RecentLabelCoveragePolicy.from_config(self.config.config)
            model_stage_ids = tuple(
                int(stage_id)
                for stage_id in self.config.config.get("canonical_events", {}).get(
                    "model_stage_ids", [1, 2, 3]
                )
            )
            coverage = require_recent_label_coverage(
                weevil_data,
                cutoff_date=today_str,
                window_years=window_years,
                stage_ids=model_stage_ids,
                policy=coverage_policy,
            )
            if not coverage.empty:
                logger.info(
                    "Recent direct-label coverage accepted before retraining: %s",
                    coverage.to_dict(orient="records"),
                )

            # Step 2: Pull historical weather data
            logger.info("Step 2: Pulling weather data via griddedweather...")
            weather_data = self.data_service.query_weather_data_for_weeviltrak_locs(
                self.config,
                today_str,
                weevil_data=weevil_data,
            )

            # Step 3: Feature engineering on weather data (train mode)
            logger.info("Step 3: Processing weather data and engineering features...")
            pw = self.data_service.process_weather_data(
                weather_data,
                config=self.config,
                weevil_data=weevil_data,
                mode="train",
            )

            # Sanity: ensure one row per (date, location_id) for training joins downstream
            # Step 3.1: Deduplicate processed weather to one row per (date, location_id)
            # Assumption: location_id -> place_id is 1:1 in processed weather.
            logger.info(
                "Step 3.1: Deduplicating processed weather (date, location_id)..."
            )
            if "date" not in pw.columns or "location_id" not in pw.columns:
                raise ValueError(
                    "Processed weather must include 'date' and 'location_id' for deduplication."
                )

            rows_before = len(pw)
            pw["date"] = pd.to_datetime(pw["date"])

            dedup_key = ["date", "location_id"]
            processed_weather_unique = (
                pw.sort_values(["location_id", "date"])
                .drop_duplicates(subset=dedup_key, keep="first")
                .copy()
            )

            rows_after = len(processed_weather_unique)
            logger.info(f"Processed weather dedup: {rows_before} -> {rows_after} rows")

            # Train set: build x_train/y_train using cutoff (today_str) and optional rolling window
            # Step 4: Prepare training data (expanding/rolling handled by window_years)
            logger.info("Step 4: Preparing training data for OOB error calculation...")
            x_train, y_train = self.data_service.prepare_training_data(
                weevil_data=weevil_data,
                weather_data=processed_weather_unique,
                test_date=today_str,
                features=self.config.features,
                target=self.config.target,
                window_years=window_years,
                post_event_training_days=int(
                    self.config.config.get(
                        "post_event_training_days",
                        self.config.config.get("post_event_zero_days", 5),
                    )
                ),
                include_test_date=include_today,
            )

            # Step 5: Train model
            logger.info("Step 5: Training model...")
            model = self.model_manager.train(x_train=x_train, y_train=y_train)

            # Optional v2.4 learned termination calibrator.  This is configured
            # only in v2.4, so v2.2/v2.3 keep their existing training behavior.
            self._train_and_save_optional_termination_model(
                x_train=x_train,
                y_train=y_train,
                cutoff_date=today_str,
                weevil_data=weevil_data,
                processed_weather=processed_weather_unique,
            )

            # Step 6: Save model to S3 (keep original behavior)
            logger.info("Step 6: Saving model to S3...")
            self.model_manager.save_model(model, model_path=model_path_override)
            logger.info("Training pipeline completed successfully!")
            logger.info(
                "Model saved to: %s",
                model_path_override or self.config.config["model_path"],
            )

        except Exception as e:
            logger.error(f"Error during training pipeline: {e}")
            raise

    def run_prediction(self, test_date: Optional[str] = None) -> pd.DataFrame:
        """
        Restartable + memory-safe prediction pipeline.

        - Load spatial coverage (place_id option B).
        - Pull raw weather filtered by coverage in 2-day chunks for [YYYY-03-01, test_date] (inclusive).
        - Cache raw weather per-day parquet (no in-memory accumulation).
        - Batch feature engineering from cached daily files -> processed_batch_*.parquet.
        - Load only target date rows -> build x_test -> predict -> format -> upload to S3.
        - Cleanup: delete outputs/cache/weather_1 and outputs/cache/processed_weather at end of run (success or failure).

        """
        if test_date is None:
            test_date = datetime.now().strftime("%Y-%m-%d")

        # Normalize/validate test date
        today = pd.to_datetime(test_date).strftime("%Y-%m-%d")
        ts_today = pd.to_datetime(today)
        year = ts_today.year
        season_start_this_year = pd.Timestamp(year=year, month=3, day=1)

        # If test_date is before Mar 1 of this year, use last season start (last year's Mar 1)
        if ts_today < season_start_this_year:
            pull_start = f"{year - 1}-03-01"
        else:
            pull_start = f"{year}-03-01"

        pull_end = today  # inclusive
        raw_cache_dir = "outputs/cache/weather_1"
        processed_cache_dir = "outputs/cache/processed_weather"

        print(
            f"Prediction pipeline for date={today} | season_start={pull_start} | weather pull range: {pull_start}..{pull_end}"
        )
        print(Path(raw_cache_dir).resolve())
        print(Path(processed_cache_dir).resolve())

        try:
            # Load: spatial coverage keys for weather filtering + enrichment columns for joins
            coverage_enrich, coverage_min = self._load_spatial_coverage_enrich()
            # Load/Cache: pull raw weather in 2-day chunks and cache to per-day parquet (restartable)
            self._cache_raw_weather_two_day_chunks(
                coverage_enrich=coverage_enrich,
                coverage_min=coverage_min,
                pull_start=pull_start,
                pull_end=pull_end,
                cache_dir="outputs/cache/weather_1",
                resolution="10by10",
            )
            # Features: batch process cached weather -> processed_batch_*.parquet (restartable)
            self._process_weather_batches_from_cache(
                coverage_enrich=coverage_enrich,
                cache_root="outputs/cache/weather_1",
                out_dir="outputs/cache/processed_weather",
                batch_size=300,
            )
            # Build the visible current-season trajectory.  The learned Ridge
            # layer depends on lag/slope/crossing features and must never be
            # asked to calibrate a one-row, no-history frame.
            x_test = self._build_x_test_for_range(
                processed_dir="outputs/cache/processed_weather",
                start_date=pull_start,
                end_date=today,
            )
            # Predict: load trained model + predict using feature columns only
            self.model_manager.load_model()
            features = self.config.features
            y_pred = self.model_manager.predict(x_test[features])

            # format output
            trajectory_out = self._format_prediction_output(
                x_test=x_test,
                y_pred=y_pred,
                today=today,
            )
            trajectory_out["trajectory_year"] = year
            trajectory_out = self._apply_optional_termination_rule(trajectory_out)
            # The S3 contract publishes only today's state, after calibration
            # has consumed the complete visible trajectory.
            out = trajectory_out[
                pd.to_datetime(trajectory_out["prediction_date"]).eq(ts_today)
            ].copy()
            out = self._apply_business_output_mapping(out)
            # save to S3
            self._save_predictions_to_s3(out=out, today=today)
        finally:
            # Cleanup caches to keep disk/memory clean across runs
            # shutil.rmtree(raw_cache_dir, ignore_errors=True)
            # shutil.rmtree(processed_cache_dir, ignore_errors=True)
            shutil.rmtree(raw_cache_dir)
            shutil.rmtree(processed_cache_dir)
            gc.collect()

    # -----------------------------
    # Step 1: spatial coverage
    # -----------------------------
    def _load_spatial_coverage_enrich(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load spatial coverage from S3 and return:
          - coverage_enrich: centroid keys + place_id + location_id/lat/lon (needed downstream)
          - coverage_min: centroid keys only (used for weather filtering)
        """
        data_source = self.config.config.get("data_source", {})
        spatial_coverage_path = data_source.get("spatial_coverage")
        if not spatial_coverage_path:
            raise ValueError("Missing config.config['data_source']['spatial_coverage']")

        spatial_coverage = self.s3_manager.read_file(spatial_coverage_path)

        coverage_enrich = (
            spatial_coverage[
                [
                    "centroid_lat",
                    "centroid_lon",
                    "place_id",
                    "location_id",
                    "latitude",
                    "longitude",
                ]
            ]
            .drop_duplicates(subset=["centroid_lat", "centroid_lon"])
            .reset_index(drop=True)
        )

        coverage_min = coverage_enrich[["centroid_lat", "centroid_lon"]].copy()

        logger.info(
            f"Loaded spatial coverage: rows={len(spatial_coverage)} | unique centroids={len(coverage_min)}"
        )
        return coverage_enrich, coverage_min

    # -----------------------------
    # Step 2: pull + cache raw weather (2-day chunks)
    # -----------------------------
    @staticmethod
    def _iter_day_chunks(
        start_ymd: str, end_ymd: str, step_days: int = 2
    ) -> Iterable[Tuple[str, str]]:
        """
        Yield inclusive date chunks for incremental weather pulls.
        Notes: Used to limit memory usage and support restartable caching.
        Output: iterator of (chunk_start, chunk_end) strings.

        Args:
            start/end (YYYY-MM-DD) + step_days.
        """
        cur = pd.Timestamp(start_ymd)
        end = pd.Timestamp(end_ymd)
        while cur <= end:
            nxt = cur + pd.Timedelta(days=step_days - 1)
            if nxt > end:
                nxt = end
            yield cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")
            cur = nxt + pd.Timedelta(days=1)

    def _cache_raw_weather_two_day_chunks(
        self,
        coverage_enrich: pd.DataFrame,
        coverage_min: pd.DataFrame,
        pull_start: str,
        pull_end: str,
        cache_dir: str,
        resolution: str = "10by10",
        step_days: int = 2,
    ) -> None:
        """
        Pull raw weather in small chunks, enrich with place/location, and cache per-day parquet (restartable).

        merge place_id/location columns (place_id B), and write per-day Parquet to disk.
        Restartable: skips if target daily files already exist.
        """
        out_dir = Path(cache_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for chunk_start, chunk_end in self._iter_day_chunks(
            pull_start, pull_end, step_days=step_days
        ):
            day_range = pd.date_range(chunk_start, chunk_end, freq="D")
            expected_files = [
                out_dir / f"weather_{d.strftime('%Y%m%d')}.parquet" for d in day_range
            ]
            if all(p.exists() for p in expected_files):
                logger.info(
                    f"Raw cache exists for {chunk_start}..{chunk_end}, skip Redshift pull."
                )
                continue

            logger.info(
                f"Pulling raw weather: {chunk_start}..{chunk_end} (chunk={step_days} days)"
            )

            # Build location-based requests from coverage centroids
            loc_requests = (
                coverage_enrich[["place_id", "centroid_lat", "centroid_lon"]]
                .rename(
                    columns={
                        "place_id": "location_id",
                        "centroid_lat": "latitude",
                        "centroid_lon": "longitude",
                    }
                )
                .drop_duplicates(subset=["location_id"])
            )

            weather_chunk = self.data_service.query_weather_data_for_weeviltrak_locs(
                self.config,
                start_date=chunk_start,
                end_date=chunk_end,
                weevil_data=loc_requests,
            )

            if weather_chunk is None or weather_chunk.empty:
                logger.warning(f"No weather returned for {chunk_start}..{chunk_end}")
                del weather_chunk
                gc.collect()
                continue

            weather_chunk = weather_chunk.copy()
            weather_chunk["date"] = pd.to_datetime(weather_chunk["date"])

            # Attach place_id/location columns (place_id option B)
            weather_chunk = weather_chunk.merge(
                coverage_enrich,
                on=["centroid_lat", "centroid_lon"],
                how="inner",
            )

            # Write one parquet per day
            for day_str, df_day in weather_chunk.groupby(
                weather_chunk["date"].dt.strftime("%Y%m%d"), sort=True
            ):
                out_file = out_dir / f"weather_{day_str}.parquet"
                if out_file.exists():
                    continue
                df_day.to_parquet(out_file, index=False)
                logger.info(f"Saved raw weather -> {out_file} | rows={len(df_day)}")

            del weather_chunk
            gc.collect()

    # -----------------------------
    # Step 3: batch feature engineering from cache
    # -----------------------------
    def _process_weather_batches_from_cache(
        self,
        coverage_enrich: pd.DataFrame,
        cache_root: str,
        out_dir: str,
        batch_size: int = 300,
    ) -> None:
        """
        Batch feature engineering from cached daily Parquet files.
        Writes processed_batch_*.parquet to disk (restartable by file existence).
        """
        cache_root_p = Path(cache_root)
        out_dir_p = Path(out_dir)
        out_dir_p.mkdir(parents=True, exist_ok=True)

        pattern = re.compile(r"weather_(\d{8})\.parquet$")
        items: List[Tuple[str, Path]] = []
        for p in cache_root_p.rglob("weather_*.parquet"):
            m = pattern.search(p.name)
            if m:
                items.append((m.group(1), p))
        if not items:
            raise FileNotFoundError(
                f"No weather_YYYYMMDD.parquet found under {cache_root_p}"
            )

        items.sort(key=lambda x: x[0])
        paths = [p for _, p in items]
        logger.info(
            f"Found {len(paths)} daily raw files: {items[0][0]} ... {items[-1][0]}"
        )

        # Columns we need for processing + downstream prepare_test_data
        keep_cols = [
            "centroid_lat",
            "centroid_lon",
            "date",
            "air_temp_max_c",
            "air_temp_min_c",
            "air_temp_avg_c",
            "soil_temp_c",
            "precip_total_mm",
            "humidity_mean_pct",
            "soil_moisture_mean_m3m3",
            "sun_duration_min",
            "place_id",
            "location_id",
            "latitude",
            "longitude",
        ]

        # Intersect with schema once (assume raw daily parquet schema consistent)
        raw_schema_cols = pq.read_schema(paths[0]).names
        raw_cols = [c for c in keep_cols if c in raw_schema_cols]

        # Unique locations from first day
        first_df = pd.read_parquet(
            paths[0], columns=["centroid_lat", "centroid_lon"]
        ).drop_duplicates()
        locs = list(first_df.itertuples(index=False, name=None))
        del first_df
        gc.collect()

        num_batches = (len(locs) + batch_size - 1) // batch_size
        logger.info(
            f"Unique locations={len(locs)} | batch_size={batch_size} | batches={num_batches}"
        )

        for i in range(0, len(locs), batch_size):
            batch_idx = i // batch_size
            out_file = out_dir_p / f"processed_batch_{batch_idx:04d}.parquet"
            if out_file.exists():
                continue  # restartable

            batch_locs = locs[i : i + batch_size]
            batch_key = pd.DataFrame(
                batch_locs, columns=["centroid_lat", "centroid_lon"]
            )

            parts: List[pd.DataFrame] = []
            for fp in paths:
                df_day = pd.read_parquet(fp, columns=raw_cols)
                sub = df_day.merge(
                    batch_key, on=["centroid_lat", "centroid_lon"], how="inner"
                )
                if not sub.empty:
                    parts.append(sub)
                del df_day, sub
            gc.collect()

            if not parts:
                del batch_key
                gc.collect()
                continue

            window = pd.concat(parts, ignore_index=True)
            window["date"] = pd.to_datetime(window["date"])

            processed = self.data_service.process_weather_data(
                window, config=self.config, mode="predict"
            )
            processed = processed.loc[:, ~processed.columns.duplicated()]

            processed.to_parquet(out_file, index=False)
            logger.info(
                f"[{batch_idx+1}/{num_batches}] saved -> {out_file} | rows={len(processed)}"
            )

            del batch_key, parts, window, processed
            gc.collect()

    # -----------------------------
    # Step 4: load processed rows for date + build x_test
    # -----------------------------
    @staticmethod
    def _load_processed_weather_for_dates_safe(
        processed_dir: str,
        start_date: str,
        end_date: Optional[str],
        requested_cols: List[str],
    ) -> pd.DataFrame:
        """
        Read processed_batch_*.parquet and return rows between [start_date, end_date].
        Only reads columns that exist on disk (schema-guard).
        """
        if end_date is None:
            end_date = start_date

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        processed_dir_p = Path(processed_dir)
        files = sorted(processed_dir_p.glob("processed_batch_*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No processed_batch_*.parquet under {processed_dir_p}"
            )

        schema_cols = pq.read_schema(files[0]).names
        cols = [c for c in requested_cols if c in schema_cols]
        if "date" not in cols:
            cols = ["date"] + cols

        parts: List[pd.DataFrame] = []
        for fp in files:
            df = pd.read_parquet(fp, columns=cols)
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
            if not df.empty:
                parts.append(df)

        if not parts:
            return pd.DataFrame(columns=cols)

        return pd.concat(parts, ignore_index=True)

    def _build_x_test_for_range(
        self, processed_dir: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        features = self.config.features
        requested = (
            ["date"] + features + ["latitude", "longitude", "place_id", "location_id"]
        )

        requested = list(dict.fromkeys(requested))  # de-duplicate

        processed_weather_small = self._load_processed_weather_for_dates_safe(
            processed_dir=processed_dir,
            start_date=start_date,
            end_date=end_date,
            requested_cols=requested,
        )

        x_test = self.data_service.prepare_test_data_latest(
            weather_data=processed_weather_small,
            features=features,
            start_date=start_date,
            end_date=end_date,
        )

        logger.info("Seasonal trajectory x_test shape: %s", x_test.shape)
        return x_test

    def _build_x_test_for_date(self, processed_dir: str, today: str) -> pd.DataFrame:
        """Backward-compatible one-date helper for callers outside production."""
        return self._build_x_test_for_range(processed_dir, today, today)

    # -----------------------------
    # Step 5/6: format + save
    # -----------------------------
    @staticmethod
    def _build_business_prediction_fields(
        prediction_date: str, y_pred
    ) -> Tuple[np.ndarray, np.ndarray]:
        raw_signed_days = np.asarray(y_pred, dtype=float)
        predicted_days = np.clip(raw_signed_days, a_min=0.0, a_max=None)
        return raw_signed_days, predicted_days

    @classmethod
    def _format_prediction_output(
        cls, x_test: pd.DataFrame, y_pred, today: str
    ) -> pd.DataFrame:
        columns = ["stage_id", "latitude", "longitude"]
        for col in ["place_id", "location_id"]:
            if col in x_test.columns:
                columns.append(col)

        out = x_test[columns].copy()
        if isinstance(x_test.index, pd.DatetimeIndex):
            prediction_dates = pd.to_datetime(x_test.index).normalize()
        else:
            prediction_dates = pd.Series(pd.Timestamp(today), index=x_test.index)
        out["prediction_date"] = prediction_dates.to_numpy()
        raw_signed_days, predicted_days = cls._build_business_prediction_fields(
            today, y_pred
        )
        out["raw_signed_days"] = raw_signed_days
        out["predicted_days"] = predicted_days

        pred_dates = pd.to_datetime(out["prediction_date"]) + pd.to_timedelta(
            predicted_days, unit="D"
        )
        out["predicted_stage_date"] = (
            pd.to_datetime(pred_dates).astype("datetime64[ns]").astype(str).str[:10]
        )
        return out

    def _save_predictions_to_s3(self, out: pd.DataFrame, today: str) -> None:
        output_root = self.config.config.get("output_path", "")
        if not output_root:
            raise ValueError("Missing config.config['output_path'] for S3 output")

        output_path = f"{output_root}/pred_data_{today}.csv"
        self.s3_manager.upload_file(out, output_path)
        logger.info(f"Saved predictions to S3: {output_path}")

    def _train_and_save_optional_termination_model(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        cutoff_date: str,
        weevil_data: pd.DataFrame,
        processed_weather: pd.DataFrame,
    ) -> None:
        settings = LearnedTerminationSettings.from_config(self.config.config)
        if not settings.enabled:
            return
        if not settings.model_path:
            raise ValueError(
                "termination_model.enabled is true but termination_model.model_path is missing"
            )

        x_train, y_train = select_aligned_training_window(
            features=x_train,
            target=y_train,
            cutoff_date=cutoff_date,
            window_years=settings.training_window_years,
        )
        logger.info(
            "Training learned termination model: alpha=%s threshold=%s "
            "window_years=%s rows=%s date_range=%s..%s",
            settings.alpha,
            settings.threshold,
            settings.training_window_years or "inherited",
            len(x_train),
            pd.to_datetime(x_train.index).min().date(),
            pd.to_datetime(x_train.index).max().date(),
        )
        train_frame = self._build_oof_termination_training_frame(
            weevil_data=weevil_data,
            processed_weather=processed_weather,
            cutoff_date=cutoff_date,
            window_years=settings.training_window_years,
        )
        if train_frame.empty:
            raise ValueError(
                "Cannot train learned termination model: no chronological "
                "out-of-fold trajectories were generated. The v2.4 Whitepaper "
                "forbids falling back to in-sample LightGBM predictions."
            )
        termination_model = train_learned_termination_model(
            training_frame=train_frame,
            target_days_remaining=train_frame["actual_days_to_event"].to_numpy(
                dtype=float
            ),
            alpha=settings.alpha,
        )
        self.s3_manager.upload_file(termination_model, settings.model_path)
        logger.info("Learned termination model saved to: %s", settings.model_path)

    def _build_oof_termination_training_frame(
        self,
        *,
        weevil_data: pd.DataFrame,
        processed_weather: pd.DataFrame,
        cutoff_date: str,
        window_years: Optional[int],
    ) -> pd.DataFrame:
        """Generate chronological OOF base trajectories for Ridge calibration.

        Every row used by Ridge is predicted by a LightGBM fit strictly before
        that row's season.  This mirrors the Whitepaper walk-forward design and
        avoids the in-sample calibration leakage that the earlier production
        implementation had.
        """
        cutoff = pd.Timestamp(cutoff_date).normalize()
        events = weevil_data.copy()
        events["stage_date"] = pd.to_datetime(events["stage_date"], errors="coerce")
        events = events[events["stage_date"].notna() & events["stage_date"].lt(cutoff)]
        events["year"] = events["stage_date"].dt.year.astype(int)
        years = sorted(events["year"].unique().tolist())
        raw_term = self.config.config.get("termination_model", {}) or {}
        start_md = str(raw_term.get("trajectory_start_month_day", "03-01"))
        end_md = str(raw_term.get("trajectory_end_month_day", "06-30"))
        frames: list[pd.DataFrame] = []

        for validation_year in years:
            fold_cutoff = pd.Timestamp(f"{validation_year}-01-01")
            prior_events = events[events["stage_date"].lt(fold_cutoff)].copy()
            validation_events = events[events["year"].eq(validation_year)].copy()
            if prior_events.empty or validation_events.empty:
                continue
            try:
                fold_x, fold_y = self.data_service.prepare_training_data(
                    weevil_data=prior_events,
                    weather_data=processed_weather,
                    test_date=fold_cutoff.strftime("%Y-%m-%d"),
                    features=self.config.features,
                    target=self.config.target,
                    window_years=window_years,
                    post_event_training_days=int(
                        self.config.config.get("post_event_training_days", 10)
                    ),
                )
            except (ValueError, KeyError) as exc:
                logger.warning(
                    "Skipping OOF termination fold %s: %s", validation_year, exc
                )
                continue
            if fold_x.empty:
                continue

            fold_manager = create_model_manager(self.config, self.s3_manager)
            fold_manager.train(x_train=fold_x, y_train=fold_y)
            trajectory_weather = self.data_service.prepare_test_data(
                weather_data=processed_weather,
                features=self.config.features,
                start_date=f"{validation_year}-{start_md}",
                end_date=f"{validation_year}-{end_md}",
            )
            if trajectory_weather.empty:
                continue
            locations = set(validation_events["location_id"].astype(str))
            trajectory_weather = trajectory_weather[
                trajectory_weather["location_id"].astype(str).isin(locations)
            ].copy()
            if trajectory_weather.empty:
                continue
            predicted = self._format_prediction_output(
                x_test=trajectory_weather,
                y_pred=fold_manager.predict(trajectory_weather[self.config.features]),
                today=f"{validation_year}-{start_md}",
            )
            predicted["trajectory_year"] = validation_year
            actual = (
                validation_events.sort_values("stage_date")
                .drop_duplicates(["location_id", "stage_id"], keep="first")[
                    ["location_id", "stage_id", "stage_date"]
                ]
                .rename(columns={"stage_date": "actual_stage_date"})
            )
            predicted["location_id"] = predicted["location_id"].astype(str)
            actual["location_id"] = actual["location_id"].astype(str)
            predicted = predicted.merge(
                actual, on=["location_id", "stage_id"], how="inner"
            )
            predicted["prediction_date"] = pd.to_datetime(predicted["prediction_date"])
            predicted["actual_stage_date"] = pd.to_datetime(
                predicted["actual_stage_date"]
            )
            predicted["actual_days_to_event"] = (
                predicted["actual_stage_date"] - predicted["prediction_date"]
            ).dt.total_seconds() / 86400.0
            frames.append(predicted)
            logger.info(
                "OOF termination fold %s: %s trajectory rows using only data before %s",
                validation_year,
                len(predicted),
                fold_cutoff.date(),
            )
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _load_optional_termination_model(self):
        settings = LearnedTerminationSettings.from_config(self.config.config)
        if not settings.enabled:
            return None, settings
        if not settings.model_path:
            raise ValueError(
                "termination_model.enabled is true but termination_model.model_path is missing"
            )
        logger.info("Loading learned termination model from: %s", settings.model_path)
        return self.s3_manager.read_file(settings.model_path), settings

    def _apply_optional_termination_rule(self, out: pd.DataFrame) -> pd.DataFrame:
        termination_model, settings = self._load_optional_termination_model()
        if termination_model is None:
            return out
        return apply_learned_termination_rule(
            predictions=out,
            model=termination_model,
            threshold=settings.threshold,
            overwrite_business_prediction=settings.overwrite_business_prediction,
            min_history_days=settings.min_history_days,
        ).assign(
            predicted_stage_date=lambda df: pd.to_datetime(
                df["predicted_stage_date"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
        )

    def _apply_business_output_mapping(self, out: pd.DataFrame) -> pd.DataFrame:
        """Apply configured Stage/Phase business output mapping.

        v2.4 directly predicts Stage 1, Phase 1, Stage 2, and Stage 3/Phase 2.
        Older configs retain their legacy Stage-only output unchanged.
        """
        if not canonical_events_enabled(self.config.config):
            return out
        return add_stage_output_names(out)


if __name__ == "__main__":
    pipeline = WeevilTrakPipeline(config_path="app/config/weeviltrak_v2.4.yml")
    # pipeline.run_training(today="2025-11-26")
    pipeline.run_prediction(test_date="2025-11-26")

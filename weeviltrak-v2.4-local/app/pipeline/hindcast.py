"""
Historical prediction (hindcast) pipeline for the configured WeevilTrak model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.pipeline.base import PipelineBase
from app.pipeline.termination import (
    LearnedTerminationSettings,
    apply_learned_termination_rule,
)
from app.settings import setup_logger

logger = setup_logger()

DEFAULT_FROZEN_MODEL_PATH = "model_testing_2025/model_weeviltrak_lgbm_v2_3.pkl"


class HindcastPipeline(PipelineBase):
    """
    Predict arbitrary historical dates using one frozen pre-trained model.
    """

    def __init__(self, config_path: str, output_dir: str = None):
        super().__init__(config_path)

        if output_dir is None:
            version = self.config.config.get("version", "unknown")
            output_dir = f"outputs/hindcast/{version}"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_default_model_path(self) -> str:
        return self.config.config.get(
            "hindcast_model_path",
            self.config.config.get("model_path", DEFAULT_FROZEN_MODEL_PATH),
        )

    @staticmethod
    def _build_prediction_fields(y_pred) -> Tuple[np.ndarray, np.ndarray]:
        raw_signed_days = np.asarray(y_pred, dtype=float)
        predicted_days = np.clip(raw_signed_days, a_min=0.0, a_max=None)
        return raw_signed_days, predicted_days

    @staticmethod
    def _normalize_location_ids(location_ids: Optional[List[str]]) -> List[str]:
        if not location_ids:
            return []
        return list(dict.fromkeys(str(location_id) for location_id in location_ids))

    def _load_spatial_coverage_enrich(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
            .drop_duplicates(subset=["centroid_lat", "centroid_lon", "location_id"])
            .reset_index(drop=True)
        )
        coverage_enrich["location_id"] = coverage_enrich["location_id"].astype("object")
        coverage_enrich["location_id"] = coverage_enrich["location_id"].where(
            coverage_enrich["location_id"].notna(), pd.NA
        )
        has_location_id = coverage_enrich["location_id"].notna()
        coverage_enrich.loc[has_location_id, "location_id"] = coverage_enrich.loc[
            has_location_id, "location_id"
        ].astype(str)
        if "place_id" in coverage_enrich.columns:
            coverage_enrich["place_id"] = coverage_enrich["place_id"].astype(str)

        coverage_min = coverage_enrich[
            ["centroid_lat", "centroid_lon"]
        ].drop_duplicates()
        return coverage_enrich, coverage_min

    @staticmethod
    def _build_custom_location_id(
        centroid_latitude: float, centroid_longitude: float
    ) -> str:
        return f"custom_gridpoint_{centroid_latitude:09.5f}_{centroid_longitude:010.5f}"

    def _map_input_locations_to_nearest_centroids(
        self,
        locations: List[Dict[str, object]],
        coverage_enrich: pd.DataFrame,
    ) -> pd.DataFrame:
        if not locations:
            return pd.DataFrame()

        input_df = pd.DataFrame(locations).copy()
        required_cols = {"latitude", "longitude"}
        missing = required_cols - set(input_df.columns)
        if missing:
            raise ValueError(
                f"Custom locations missing required field(s): {sorted(missing)}"
            )

        input_df["input_name"] = input_df.get("name")
        input_df["input_latitude"] = pd.to_numeric(
            input_df["latitude"], errors="coerce"
        )
        input_df["input_longitude"] = pd.to_numeric(
            input_df["longitude"], errors="coerce"
        )
        if input_df[["input_latitude", "input_longitude"]].isna().any().any():
            raise ValueError("Custom locations must provide numeric latitude/longitude")

        centroids = (
            coverage_enrich[
                [
                    "centroid_lat",
                    "centroid_lon",
                    "location_id",
                    "place_id",
                    "latitude",
                    "longitude",
                ]
            ]
            .drop_duplicates(subset=["centroid_lat", "centroid_lon", "location_id"])
            .reset_index(drop=True)
        )
        centroid_lat = centroids["centroid_lat"].astype(float).to_numpy()
        centroid_lon = centroids["centroid_lon"].astype(float).to_numpy()

        rows: List[Dict[str, object]] = []
        for row in input_df.itertuples(index=False):
            dist = (centroid_lat - float(row.input_latitude)) ** 2 + (
                centroid_lon - float(row.input_longitude)
            ) ** 2
            idx = int(np.argmin(dist))
            match = centroids.iloc[idx]
            matched_centroid_lat = float(match["centroid_lat"])
            matched_centroid_lon = float(match["centroid_lon"])
            source_location_id = (
                str(match["location_id"]) if pd.notna(match["location_id"]) else None
            )
            resolved_location_id = source_location_id or self._build_custom_location_id(
                matched_centroid_lat,
                matched_centroid_lon,
            )
            request_latitude = (
                float(match["latitude"])
                if pd.notna(match["latitude"])
                else matched_centroid_lat
            )
            request_longitude = (
                float(match["longitude"])
                if pd.notna(match["longitude"])
                else matched_centroid_lon
            )
            rows.append(
                {
                    "input_name": row.input_name,
                    "input_latitude": float(row.input_latitude),
                    "input_longitude": float(row.input_longitude),
                    "mapped_centroid_latitude": matched_centroid_lat,
                    "mapped_centroid_longitude": matched_centroid_lon,
                    "source_location_id": source_location_id,
                    "location_id": resolved_location_id,
                    "place_id": (
                        str(match["place_id"]) if pd.notna(match["place_id"]) else None
                    ),
                    "latitude": request_latitude,
                    "longitude": request_longitude,
                }
            )

        mapped = pd.DataFrame(rows)
        return mapped

    def _resolve_prediction_scope(
        self,
        location_ids: Optional[List[str]],
        locations: Optional[List[Dict[str, object]]],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        coverage_enrich, _ = self._load_spatial_coverage_enrich()

        normalized_ids = self._normalize_location_ids(location_ids)
        coverage_selected = pd.DataFrame()
        if normalized_ids:
            coverage_selected = coverage_enrich[
                coverage_enrich["location_id"].isin(normalized_ids)
            ].copy()

        mapped_locations = self._map_input_locations_to_nearest_centroids(
            locations or [],
            coverage_enrich,
        )

        if not normalized_ids and mapped_locations.empty:
            scope = coverage_enrich.dropna(
                subset=["location_id", "latitude", "longitude"]
            ).copy()
            resolved_ids = sorted(scope["location_id"].astype(str).unique().tolist())
        else:
            scope_parts = []
            if not coverage_selected.empty:
                scope_parts.append(coverage_selected)
            if not mapped_locations.empty:
                mapped_scope = mapped_locations[
                    [
                        "mapped_centroid_latitude",
                        "mapped_centroid_longitude",
                        "place_id",
                        "location_id",
                        "latitude",
                        "longitude",
                    ]
                ].rename(
                    columns={
                        "mapped_centroid_latitude": "centroid_lat",
                        "mapped_centroid_longitude": "centroid_lon",
                    }
                )
                scope_parts.append(mapped_scope)

            scope = (
                pd.concat(scope_parts, ignore_index=True)
                if scope_parts
                else pd.DataFrame()
            )
            scope = scope.dropna(subset=["location_id", "latitude", "longitude"]).copy()
            scope["location_id"] = scope["location_id"].astype(str)
            resolved_ids = sorted(
                set(scope["location_id"].astype(str).tolist()) | set(normalized_ids)
            )

        if scope.empty:
            raise ValueError(
                "No prediction locations resolved from the provided inputs."
            )

        scope = scope.drop_duplicates(subset=["location_id"]).reset_index(drop=True)
        return scope, mapped_locations, resolved_ids

    def _prepare_weather_for_dates(
        self,
        scope_locations: pd.DataFrame,
        target_dates: List[str],
    ) -> pd.DataFrame:
        target_ts = sorted(pd.to_datetime(target_dates))
        start_date = target_ts[0].strftime("%Y-%m-%d")
        end_date = target_ts[-1].strftime("%Y-%m-%d")

        weather_raw = self.data_service.query_weather_data_for_weeviltrak_locs(
            config=self.config,
            end_date=end_date,
            start_date=start_date,
            weevil_data=scope_locations[
                ["location_id", "latitude", "longitude"]
            ].copy(),
        )
        processed_weather = self.data_service.process_weather_data(
            weather_data=weather_raw,
            config=self.config,
            mode="predict",
        )

        pw = processed_weather.copy()
        pw["date"] = pd.to_datetime(pw["date"])
        pw["location_id"] = pw["location_id"].astype(str)
        pw = (
            pw.sort_values(["location_id", "date"])
            .drop_duplicates(subset=["date", "location_id"], keep="first")
            .reset_index(drop=True)
        )
        return pw

    def _predict_for_dates(
        self,
        processed_weather_unique: pd.DataFrame,
        target_dates: List[str],
        stages: List[int],
        resolved_location_ids: List[str],
        mapped_locations: pd.DataFrame,
    ) -> pd.DataFrame:
        outputs: List[pd.DataFrame] = []
        mapped_meta = mapped_locations.drop_duplicates(subset=["location_id"]).copy()

        for target_date in sorted(
            pd.to_datetime(target_dates).strftime("%Y-%m-%d").tolist()
        ):
            x_test = self.data_service.prepare_test_data_latest(
                weather_data=processed_weather_unique,
                features=self.config.features,
                start_date=target_date,
                end_date=target_date,
            )
            if x_test.empty:
                continue

            x_test = x_test[
                x_test["location_id"].astype(str).isin(resolved_location_ids)
            ].copy()
            x_test = x_test[x_test["stage_id"].isin(stages)].copy()
            if x_test.empty:
                continue

            y_pred = self.model_manager.predict(x_test[self.config.features])
            raw_signed_days, predicted_days = self._build_prediction_fields(y_pred)

            base_cols = ["stage_id", "latitude", "longitude"]
            for col in ["place_id", "location_id"]:
                if col in x_test.columns and col not in base_cols:
                    base_cols.append(col)

            out = x_test[base_cols].copy()
            out["prediction_date"] = target_date
            out["raw_signed_days"] = raw_signed_days
            out["predicted_days"] = predicted_days
            pred_dates = pd.to_datetime(target_date) + pd.to_timedelta(
                predicted_days, unit="D"
            )
            out["predicted_stage_date"] = pd.to_datetime(pred_dates).normalize()

            if not mapped_meta.empty:
                out = out.merge(
                    mapped_meta[
                        [
                            "location_id",
                            "input_name",
                            "input_latitude",
                            "input_longitude",
                            "mapped_centroid_latitude",
                            "mapped_centroid_longitude",
                        ]
                    ],
                    on="location_id",
                    how="left",
                )

            outputs.append(out)

        if not outputs:
            return pd.DataFrame()

        combined = pd.concat(outputs, ignore_index=True)
        combined["prediction_date"] = pd.to_datetime(combined["prediction_date"])
        combined["predicted_stage_date"] = pd.to_datetime(
            combined["predicted_stage_date"]
        )
        combined = self._apply_optional_termination_rule(combined)
        combined["predicted_stage_date"] = pd.to_datetime(
            combined["predicted_stage_date"], errors="coerce"
        )
        return combined

    def _apply_optional_termination_rule(self, out: pd.DataFrame) -> pd.DataFrame:
        settings = LearnedTerminationSettings.from_config(self.config.config)
        if not settings.enabled or out is None or out.empty:
            return out
        if not settings.model_path:
            raise ValueError(
                "termination_model.enabled is true but termination_model.model_path is missing"
            )
        termination_model = self.s3_manager.read_file(settings.model_path)
        return apply_learned_termination_rule(
            predictions=out,
            model=termination_model,
            threshold=settings.threshold,
            overwrite_business_prediction=settings.overwrite_business_prediction,
            min_history_days=settings.min_history_days,
        )

    @staticmethod
    def _compute_pest_year(prediction_date: pd.Series) -> pd.Series:
        pred = pd.to_datetime(prediction_date)
        return np.where(pred.dt.month >= 3, pred.dt.year, pred.dt.year - 1)

    def _join_actuals(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        if pred_df.empty:
            return pred_df

        max_pred_date = pred_df["prediction_date"].max()
        weevil_data = self.data_service.pull_weevil_data(
            today=max_pred_date.strftime("%Y-%m-%d")
        )
        if weevil_data.empty:
            pred_df["actual_stage_date"] = pd.NaT
            pred_df["error_days"] = np.nan
            return pred_df

        actual = weevil_data.copy()
        actual["stage_date"] = pd.to_datetime(actual["stage_date"])
        actual["location_id"] = actual["location_id"].astype(str)
        actual["stage_id"] = pd.to_numeric(actual["stage_id"], errors="coerce").astype(
            int
        )
        actual["pest_year"] = np.where(
            actual["stage_date"].dt.month >= 3,
            actual["stage_date"].dt.year,
            actual["stage_date"].dt.year - 1,
        )

        pred = pred_df.copy()
        pred["location_id"] = pred["location_id"].astype(str)
        pred["stage_id"] = pd.to_numeric(pred["stage_id"], errors="coerce").astype(int)
        pred["pest_year"] = self._compute_pest_year(pred["prediction_date"])
        pred["_prediction_row_id"] = np.arange(len(pred))

        merged = pred.merge(
            actual[["location_id", "stage_id", "pest_year", "stage_date"]],
            on=["location_id", "stage_id", "pest_year"],
            how="left",
        )
        merged = merged[
            merged["stage_date"].isna()
            | (merged["stage_date"] >= merged["prediction_date"])
        ].copy()
        merged["days_to_actual"] = (
            merged["stage_date"] - merged["prediction_date"]
        ).dt.days
        merged = merged.sort_values(
            ["_prediction_row_id", "days_to_actual", "stage_date"],
            na_position="last",
        )
        best_actual = merged.drop_duplicates(
            subset=["_prediction_row_id"], keep="first"
        )
        best_actual = best_actual[["_prediction_row_id", "stage_date"]].rename(
            columns={"stage_date": "actual_stage_date"}
        )

        out = pred.merge(best_actual, on="_prediction_row_id", how="left")
        out["error_days"] = (
            out["predicted_stage_date"] - out["actual_stage_date"]
        ).dt.days
        out = out.drop(columns=["_prediction_row_id", "pest_year"])
        return out

    def _save_local_outputs(self, out: pd.DataFrame) -> None:
        timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_dir / f"hindcast_detail_{timestamp}.csv"
        parquet_path = self.output_dir / f"hindcast_detail_{timestamp}.parquet"
        out_to_save = out.copy()
        if "prediction_date" in out_to_save.columns:
            out_to_save["prediction_date"] = pd.to_datetime(
                out_to_save["prediction_date"]
            ).dt.strftime("%Y-%m-%d")
        for col in ["predicted_stage_date", "actual_stage_date"]:
            if col in out_to_save.columns:
                out_to_save[col] = pd.to_datetime(
                    out_to_save[col], errors="coerce"
                ).dt.strftime("%Y-%m-%d")
        out_to_save.to_csv(csv_path, index=False)
        out_to_save.to_parquet(parquet_path, index=False)
        logger.info("Saved hindcast outputs to %s and %s", csv_path, parquet_path)

    def run_hindcast(
        self,
        target_dates: List[str],
        stages: List[int] = [1, 2, 3],
        location_ids: List[str] | None = None,
        locations: List[Dict[str, object]] | None = None,
        join_actuals: bool = False,
        model_path_override: str | None = None,
        output_dir: str | None = None,
        save_local: bool = True,
    ) -> pd.DataFrame:
        if not target_dates:
            raise ValueError("target_dates must not be empty")

        if output_dir is not None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        scope_locations, mapped_locations, resolved_location_ids = (
            self._resolve_prediction_scope(
                location_ids=location_ids,
                locations=locations,
            )
        )
        model_path = model_path_override or self.get_default_model_path()
        self.model_manager.load_model(model_path=model_path)

        processed_weather_unique = self._prepare_weather_for_dates(
            scope_locations=scope_locations,
            target_dates=target_dates,
        )
        out = self._predict_for_dates(
            processed_weather_unique=processed_weather_unique,
            target_dates=target_dates,
            stages=stages,
            resolved_location_ids=resolved_location_ids,
            mapped_locations=mapped_locations,
        )
        if join_actuals:
            out = self._join_actuals(out)

        if save_local and not out.empty:
            self._save_local_outputs(out)

        return out


def _parse_locations_arg(location_values: List[str]) -> List[Dict[str, object]]:
    parsed: List[Dict[str, object]] = []
    for raw in location_values:
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) not in {2, 3}:
            raise ValueError(
                "--location entries must use 'lat,lon' or 'name,lat,lon' format"
            )
        if len(parts) == 2:
            lat_s, lon_s = parts
            name = None
        else:
            name, lat_s, lon_s = parts
        parsed.append(
            {
                "name": name,
                "latitude": float(lat_s),
                "longitude": float(lon_s),
            }
        )
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WeevilTrak historical hindcast.")
    parser.add_argument("--config-path", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--target-date",
        dest="target_dates",
        action="append",
        required=True,
        help="Historical prediction date (repeat for multiple dates).",
    )
    parser.add_argument(
        "--stage",
        dest="stages",
        action="append",
        type=int,
        help="Stage id to predict (repeatable, defaults to 1/2/3).",
    )
    parser.add_argument(
        "--location-id",
        dest="location_ids",
        action="append",
        help="Existing location_id to include (repeatable).",
    )
    parser.add_argument(
        "--location",
        dest="locations",
        action="append",
        default=[],
        help="Custom location in 'lat,lon' or 'name,lat,lon' format (repeatable).",
    )
    parser.add_argument(
        "--join-actuals", action="store_true", help="Join next actual stage date."
    )
    parser.add_argument(
        "--model-path-override", help="Override model path for this run."
    )
    parser.add_argument("--output-dir", help="Optional local output directory.")
    parser.add_argument(
        "--no-save-local",
        action="store_true",
        help="Do not save local CSV/Parquet outputs.",
    )
    args = parser.parse_args()

    pipeline = HindcastPipeline(
        config_path=args.config_path, output_dir=args.output_dir
    )
    locations = _parse_locations_arg(args.locations)
    out = pipeline.run_hindcast(
        target_dates=args.target_dates,
        stages=args.stages or [1, 2, 3],
        location_ids=args.location_ids,
        locations=locations,
        join_actuals=args.join_actuals,
        model_path_override=args.model_path_override,
        output_dir=args.output_dir,
        save_local=not args.no_save_local,
    )
    print(out.head())


if __name__ == "__main__":
    main()

"""
Data processing service for managing data from the database and S3 bucket.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt

import os
import shutil
from typing import Optional, List, Tuple, Literal
from datetime import datetime, timedelta
from app.services.database_service import DatabaseManager
from app.services.canonical_events import (
    CanonicalEventConfig,
    append_phase1_midpoint_training_events,
    build_canonical_training_events,
    model_stage_ids_from_config,
)
from app.services.manual_events import (
    ManualEventConfig,
    append_manual_stage_events,
)
from griddedweather.s3_store import S3Manager
from griddedweather import GriddedWeatherClient, WeatherConfig, RedshiftCredentials
from app.config.weevilltrak_config import WeevillTrakConfig
from app.settings import setup_logger

logger = setup_logger()

# Column aliases: raw Redshift -> short names used downstream
COLUMN_ALIASES = {
    "temperature_c_2_m_above_gnd_max": "air_temp_max_c",
    "temperature_c_2_m_above_gnd_min": "air_temp_min_c",
    "temperature_c_2_m_above_gnd_avg": "air_temp_avg_c",
    "temperature_c_0to10cm_below_gnd_avg": "soil_temp_c",
    "precipitation_total_grid_mm_surface_sum": "precip_total_mm",
    "relative_humidity_pct_2_m_above_gnd_avg": "humidity_mean_pct",
    "soil_water_content_m3_per_m3_0to10cm_below_gnd_avg": "soil_moisture_mean_m3m3",
    "daylight_duration_min_surface_sum": "sun_duration_min",
}


class DataPreparationService:
    """
    Service for processing data from the database and S3 bucket.
    Handles weevil data extraction, weather data processing, and feature engineering.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        s3_manager: S3Manager,
        weather_client: Optional[GriddedWeatherClient] = None,
        s3_bucket: Optional[str] = None,
    ):
        """
        Initialize data processing service.

        Args:
            db_manager: Database manager for accessing Redshift
            s3_manager: S3 manager for accessing stored data
            weather_client: Optional griddedweather client. If not supplied, one is
                created using default environment-driven configuration (same env as
                db_manager / s3_manager).
            s3_bucket: S3 bucket forwarded to WeatherConfig when constructing the
                griddedweather client automatically (if provided).
        """
        self.db_manager = db_manager
        self.s3_manager = s3_manager
        # Use the same env-driven configuration used by db_manager / s3_manager;
        # do not reconfigure if a client is already provided.
        self.weather_client = weather_client or GriddedWeatherClient(
            self._create_weather_config(s3_bucket)
        )
        logger.info("DataPreparationService initialized")

    def pull_weevil_data(
        self,
        today: str,
        outliers: Optional[List[str]] = None,
        query: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Retrieve and process weevil data from Redshift.
        Processed weevil data includes:
        - Stage date
        - Latitude
        - Longitude
        - Location ID
        - Year
        - Day of year
        - Stage ID
        - Stage name

        Args:
            today: Cutoff date for filtering weevil data
            outliers: List of location_ids to exclude

        Returns:
            Processed weevil data DataFrame
        """
        end_date = today

        logger.info("Pulling weevil data from Redshift...")

        if query is None:
            query = "SELECT * FROM europe_dna.europe_dna_sps_weeviltrak"

        with self.db_manager as db:
            conn = db.get_connection()

            weevil_data = pd.read_sql(query, conn)

        logger.info(f"Weevil data: {weevil_data.head()}")
        # Parse dates and filter
        weevil_data["stage_date"] = pd.to_datetime(weevil_data["stage_date"])
        weevil_data = weevil_data[weevil_data["stage_date"] <= pd.to_datetime(end_date)]

        # Type conversions
        weevil_data["latitude"] = weevil_data["latitude"].astype(float)
        weevil_data["longitude"] = weevil_data["longitude"].astype(float)
        weevil_data["location_id"] = weevil_data["location_id"].astype(str)
        weevil_data["year"] = weevil_data["stage_date"].dt.year.astype("int64")
        weevil_data["day_of_year"] = weevil_data["stage_date"].dt.dayofyear

        canonical_settings = CanonicalEventConfig.from_config(
            self.db_manager.config.config
            if self.db_manager.config is not None
            else None
        )
        if canonical_settings.enabled:
            weevil_data = build_canonical_training_events(
                weevil_data,
                config=canonical_settings,
            )
        else:
            # Filter for valid stages and clean up stage_id
            valid_stages = {f"Stage {i}" for i in range(1, 6)}
            weevil_data = weevil_data[weevil_data["stage_name"].isin(valid_stages)]
            weevil_data.drop(columns=["stage_id"], inplace=True)
            weevil_data = weevil_data.rename(columns={"stage_name": "stage_id"})
            weevil_data["stage_id"] = weevil_data["stage_id"].apply(
                lambda x: int(x.replace("Stage ", ""))
            )

        manual_settings = ManualEventConfig.from_config(
            self.db_manager.config.config
            if self.db_manager.config is not None
            else None
        )
        if manual_settings.enabled:
            before_manual = len(weevil_data)
            weevil_data = append_manual_stage_events(
                weevil_data,
                config=manual_settings,
            )
            weevil_data = weevil_data[
                weevil_data["stage_date"] <= pd.to_datetime(end_date)
            ]
            added_manual = int(weevil_data.get("is_manual_event", False).sum())
            logger.info(
                "Manual curated event injection enabled: %d Redshift/canonical rows + %d manual rows",
                before_manual,
                added_manual,
            )

        # Phase 1 is a direct v2.4 model target.  Midpoint labels are built
        # after curated manual rows have been appended so all eligible Stage
        # 1/2 pairs are treated consistently.  Release configs may also retain
        # quality-controlled observed Phase 1 rows as canonical target ID 4;
        # this does not infer Stage 1/2 labels from Phase observations.
        if canonical_settings.enabled:
            weevil_data = append_phase1_midpoint_training_events(
                weevil_data,
                config=canonical_settings,
            )

        # Remove duplicates and outliers
        weevil_data = weevil_data.drop_duplicates(
            subset=["latitude", "longitude", "year", "stage_id"], keep="first"
        )
        if outliers:
            weevil_data = weevil_data[~weevil_data["location_id"].isin(outliers)]

        # Remove known bad records (data quality issue in source table)
        bad_location_year_pairs = {
            ("85", 2024),
            ("88", 2023),
            ("61", 2023),
        }

        weevil_data = weevil_data[
            ~weevil_data.apply(
                lambda r: (r["location_id"], int(r["year"])) in bad_location_year_pairs,
                axis=1,
            )
        ]

        # Keep only records between March 1 and June 1 (inclusive)
        manual_mask = (
            weevil_data.get("is_manual_event", False)
            if "is_manual_event" in weevil_data.columns
            else False
        )
        weevil_data = weevil_data[
            (
                (weevil_data["day_of_year"] >= 60)
                & (weevil_data["day_of_year"] <= 152)  # March 1  # June 1
            )
            | manual_mask
        ]

        weevil_data.sort_values(by=["year", "stage_id"], inplace=True)

        # Remove Canadian location (Gulph) for now
        weevil_data = weevil_data[weevil_data.location_id != "5"]

        logger.info(f"Successfully processed {len(weevil_data)} weevil records")
        return weevil_data

    def query_weather_data_for_weeviltrak_locs(
        self,
        config: WeevillTrakConfig,
        end_date: str,
        start_date: Optional[str] = None,
        weevil_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Query historical 10x10 km weather for WeevilTrak locations via griddedweather.

        Primary path (preferred):
            - Build location-based requests from ``weevil_data`` (unique location_id, lat, lon).
            - Infer per-location start/end to span March 1 to June 1 for the years with weevil records.
            - griddedweather resolves each point to the nearest centroid and returns weather.

        Args:
            config: Configuration object.
            end_date: Deprecated; kept for API compatibility (ignored when weevil_data provided).
            start_date: Deprecated; kept for API compatibility (ignored when weevil_data provided).
            weevil_data: Required for the preferred path; must include ``location_id``,
                ``latitude``, ``longitude``, and ``stage_date`` (datetime-like).

        Returns:
            Weather data DataFrame.
        """
        logger.info("Pulling weather data via griddedweather (direct Redshift pull)")

        if weevil_data is None:
            raise ValueError(
                "weevil_data is required to infer start/end date ranges for weather pulls"
            )

        wd = weevil_data.copy()
        required_loc_cols = {"location_id", "latitude", "longitude"}
        missing_loc = required_loc_cols - set(wd.columns)
        if missing_loc:
            raise ValueError(
                f"weevil_data missing required columns: {', '.join(sorted(missing_loc))}"
            )

        if "stage_date" in wd.columns:
            loc_requests = self._build_location_year_ranges(wd)
        else:
            if start_date is None or end_date is None:
                raise ValueError(
                    "weevil_data without stage_date requires explicit start_date and end_date"
                )
            loc_requests = self._build_location_date_ranges(
                location_data=wd,
                start_date=start_date,
                end_date=end_date,
            )
        bounds_map = (
            loc_requests.reset_index()
            .set_index("location_year_key")[["start_date", "end_date"]]
            .to_dict(orient="index")
        )

        df = self.weather_client.pull_weather_data(loc_requests)
        df = self._normalize_weather_columns(df)

        if df.empty:
            logger.warning("No weather data returned from griddedweather")
            return df

        # The griddedweather library merges location_id via centroid_lat/lon
        # floats, which fails due to precision mismatches.  Rebuild the
        # place_id -> location_id mapping ourselves using nearest-centroid
        # matching on the actual returned place centroids.
        if "place_id" in df.columns:
            df = self._remap_location_id(df, loc_requests)

        # If location_id/year key is only present in the index, materialize it as columns
        if "location_id" not in df.columns and df.index.name == "location_year_key":
            df = df.reset_index()
        if "location_year_key" in df.columns:
            df[["location_id", "pull_year"]] = df["location_year_key"].str.split(
                "__", 1, expand=True
            )

        if "location_id" not in df.columns and df.index.name == "location_id":
            df = df.reset_index()

        if "location_id" in df.columns:
            df["location_id"] = df["location_id"].astype(str)

        if "location_year_key" not in df.columns and {"location_id", "date"} <= set(
            df.columns
        ):
            df["location_year_key"] = df.apply(
                lambda r: f"{r['location_id']}__{pd.to_datetime(r['date']).year}",
                axis=1,
            )

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        # Per-location/year trimming to requested bounds
        if "date" in df.columns:

            def _in_bounds(row):
                if pd.isna(row.get("date")):
                    return False
                key = row.get("location_year_key")
                if not key and {"location_id", "date"} <= set(row.index):
                    key = f"{row.get('location_id', '')}__{row['date'].year}"
                b = bounds_map.get(key) if key else None
                if not b:
                    return False
                return (
                    pd.to_datetime(b["start_date"])
                    <= row["date"]
                    <= pd.to_datetime(b["end_date"])
                )

            df = df[df.apply(_in_bounds, axis=1)]

        # Drop helper column if present
        if "location_year_key" in df.columns:
            df = df.drop(columns=["location_year_key"])

        if "place_id" in df.columns:
            df["place_id"] = df["place_id"].astype(str)

        if "location_id" in df.columns:
            df["location_id"] = df["location_id"].astype(str)

        sort_cols = [
            col for col in ["place_id", "location_id", "date"] if col in df.columns
        ]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)

        logger.info(
            f"Successfully pulled {len(df)} weather records from griddedweather"
        )
        return df

    @staticmethod
    def _remap_location_id(
        df: pd.DataFrame,
        loc_requests: pd.DataFrame,
    ) -> pd.DataFrame:
        """Re-assign ``location_id`` to weather rows using place_id only.

        The griddedweather library tries to merge location_id back via
        (place_id, centroid_lat, centroid_lon), which silently fails when
        floating-point coordinates differ between the centroid lookup and
        the actual weather data.  This method builds a robust place_id →
        location_id(s) mapping by matching each requested lat/lon to the
        nearest place-centroid present in the data, then merges on
        ``place_id`` alone.
        """
        centroids = (
            df[["place_id", "centroid_lat", "centroid_lon"]]
            .drop_duplicates(subset=["place_id"])
            .copy()
        )
        if centroids.empty:
            return df

        # Unique location → lat/lon from the request
        req = loc_requests.reset_index()
        loc_coords = (
            req[["location_id", "lat", "lon"]]
            .drop_duplicates(subset=["location_id"])
            .copy()
        )
        loc_coords["location_id"] = loc_coords["location_id"].astype(str)

        # Vectorized nearest-centroid matching (haversine is unnecessary at
        # this resolution — Euclidean on degrees is sufficient for < 0.2°)
        c_lat = centroids["centroid_lat"].astype(float).to_numpy()
        c_lon = centroids["centroid_lon"].astype(float).to_numpy()
        c_pid = centroids["place_id"].astype(str).to_numpy()

        rows = []
        for _, row in loc_coords.iterrows():
            dist = (c_lat - float(row["lat"])) ** 2 + (c_lon - float(row["lon"])) ** 2
            idx = int(np.argmin(dist))
            rows.append(
                {"place_id": c_pid[idx], "location_id": str(row["location_id"])}
            )

        pid_map = pd.DataFrame(rows).drop_duplicates()

        # Drop the unreliable location_id from the library and merge our own
        df = df.drop(columns=["location_id"], errors="ignore")
        df["place_id"] = df["place_id"].astype(str)
        df = df.merge(pid_map, on="place_id", how="left")

        n_mapped = df["location_id"].notna().sum()
        logger.info(
            "Remapped location_id via place_id: %d/%d rows mapped (%d unique locations)",
            n_mapped,
            len(df),
            pid_map["location_id"].nunique(),
        )
        return df

    def _create_weather_config(self, s3_bucket: Optional[str] = None) -> WeatherConfig:
        """Construct WeatherConfig using griddedweather cache defaults while preserving Redshift creds."""
        cfg = WeatherConfig.from_env(env=os.environ)

        cfg.redshift = RedshiftCredentials(
            host=os.getenv("DATABASE_HOST", os.getenv("REDSHIFT_HOST", "")),
            port=int(os.getenv("DATABASE_PORT", os.getenv("REDSHIFT_PORT", "5439"))),
            database=os.getenv("DATABASE_NAME", ""),
            user=os.getenv("REDSHIFT_USER", ""),
            password=os.getenv("REDSHIFT_PASSWORD", ""),
        )

        if s3_bucket:
            cfg.s3_bucket = s3_bucket

        return cfg

    def _normalize_weather_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename raw weather columns and keep only fields used downstream."""
        df = df.rename(columns=COLUMN_ALIASES)

        keep_cols = [
            # keys / coords
            "location_id",
            "place_id",
            "location_year_key",
            "date",
            "centroid_lat",
            "centroid_lon",
            "matched_lat",
            "matched_lon",
            # weather features
            "air_temp_max_c",
            "air_temp_min_c",
            "air_temp_avg_c",
            "soil_temp_c",
            "precip_total_mm",
            "humidity_mean_pct",
            "soil_moisture_mean_m3m3",
            "sun_duration_min",
        ]

        cols_present = [c for c in keep_cols if c in df.columns]
        if cols_present:
            df = df[cols_present]
        return df

    def _build_location_date_ranges(
        self,
        location_data: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Build a single explicit date window per location for prediction-time pulls."""
        ld = location_data.copy()
        ld["location_id"] = ld["location_id"].astype(str)
        ld["latitude"] = pd.to_numeric(ld["latitude"], errors="coerce")
        ld["longitude"] = pd.to_numeric(ld["longitude"], errors="coerce")

        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        req_df = (
            ld[["location_id", "latitude", "longitude"]]
            .drop_duplicates(subset=["location_id"])
            .rename(columns={"latitude": "lat", "longitude": "lon"})
            .copy()
        )
        req_df["start_date"] = start_ts.strftime("%Y-%m-%d")
        req_df["end_date"] = end_ts.strftime("%Y-%m-%d")
        req_df["pull_year"] = start_ts.year
        req_df["location_year_key"] = (
            req_df["location_id"] + "__" + req_df["pull_year"].astype(str)
        )
        if not req_df.empty:
            req_df = req_df.set_index("location_id")
            req_df.index.name = "location_id"
        return req_df

    def _build_location_year_ranges(self, weevil_data: pd.DataFrame) -> pd.DataFrame:
        """Build per-location, per-year date windows (Mar 1 -> Jun 1) for weather pulls."""
        wd = weevil_data.copy()
        wd["stage_date"] = pd.to_datetime(wd["stage_date"])
        wd["year"] = wd["stage_date"].dt.year

        years_per_loc = wd.groupby("location_id")["year"].unique()

        loc_records: List[dict] = []
        coords = (
            wd[["location_id", "latitude", "longitude"]]
            .drop_duplicates(subset=["location_id"])
            .set_index("location_id")
        )

        for loc_id, years in years_per_loc.items():
            if loc_id not in coords.index:
                continue
            lat = coords.at[loc_id, "latitude"]
            lon = coords.at[loc_id, "longitude"]
            for y in years:
                loc_records.append(
                    {
                        "location_id": str(loc_id),
                        "lat": lat,
                        "lon": lon,
                        "start_date": f"{int(y)}-03-01",
                        "end_date": f"{int(y)}-06-01",
                        "location_year_key": f"{loc_id}__{int(y)}",
                        "pull_year": int(y),
                    }
                )

        req_df = pd.DataFrame(loc_records)
        if not req_df.empty:
            req_df["location_id"] = req_df["location_id"].astype(str)
            req_df = req_df.set_index("location_id")
            req_df.index.name = "location_id"
        return req_df

    def process_weather_data(
        self,
        weather_data: pd.DataFrame,
        config: Optional["WeevillTrakConfig"] = None,
        weevil_data: Optional[pd.DataFrame] = None,
        mode: Literal["train", "predict"] = "train",
    ) -> pd.DataFrame:
        """
        Process weather data with feature engineering.

        - In `train` mode: optionally maps place_id -> location_id and filters to matched locations.
        - In `predict` mode: keeps ALL place_id rows (no location_id matching / no dropna filtering).
        """
        logger.info(f"Processing weather data (mode={mode})...")

        df = weather_data.copy()

        # --- Normalize lat/lon column names to model expectations ---
        if "centroid_lat" in df.columns:
            df = df.rename(columns={"centroid_lat": "latitude"})
        if "matched_lat" in df.columns:
            df = df.rename(columns={"matched_lat": "latitude"})
        if "centroid_lon" in df.columns:
            df = df.rename(columns={"centroid_lon": "longitude"})
        if "matched_lon" in df.columns:
            df = df.rename(columns={"matched_lon": "longitude"})

        # --- Types / ordering ---
        df["date"] = pd.to_datetime(df["date"])

        # stable ordering before downstream grouped ops
        if "place_id" in df.columns:
            df = df.sort_values(["place_id", "date"])
        else:
            df = df.sort_values(["date"])

        df["precip_total_mm"] = df["precip_total_mm"].astype(float)
        df["humidity_mean_pct"] = df["humidity_mean_pct"].astype(float)

        # --- Convert temperature to Fahrenheit (as original) ---
        temp_cols = ["air_temp_max_c", "air_temp_min_c"]
        for col in temp_cols:
            f_col = col.replace("_c", "_f")
            df[f_col] = self._convert_celsius_to_fahrenheit(df[col].astype(float))

        # --- Growing Degree Days (as original; base_temp=50°F) ---
        df["gdd_air"] = self._calc_growing_degree_days(
            df["air_temp_min_f"],
            df["air_temp_max_f"],
            base_temp=50,
        )
        mean_temp_f = (df["air_temp_min_f"] + df["air_temp_max_f"]) / 2
        df["cdd_air"] = (50 - mean_temp_f).clip(lower=0)
        df["doy"] = df["date"].dt.dayofyear
        # --- Cumulative + rolling (as original) ---
        df = self._calc_annual_cumulative(df, start_month=3, start_day=1)
        rolling_window_days = 7
        if config is not None:
            rolling_window_days = int(config.config.get("rolling_window_days", 7))
        df = self._calc_rolling_avg(df, windows=rolling_window_days)

        # --- Train-only: map place_id to weevil location_id and filter to matched ---
        if mode == "train":
            if (
                config is not None
                and weevil_data is not None
                and "location_id" not in df.columns
            ):
                logger.info("Matching weather place_ids to weevil location_ids...")

                data_source = config.config.get("data_source", {})
                spatial_coverage_path = data_source.get("spatial_coverage")

                if spatial_coverage_path:
                    spatial_coverage = self.s3_manager.read_file(spatial_coverage_path)

                    weevil_locations = weevil_data[
                        ["location_id", "latitude", "longitude"]
                    ].drop_duplicates()

                    matched_locations = self.match_locations_to_place_id(
                        weevil_locations,
                        spatial_coverage,
                    )

                    location_place_mapping = matched_locations[
                        ["location_id", "place_id"]
                    ].dropna()

                    df = df.merge(location_place_mapping, on="place_id", how="left")

                    # IMPORTANT: training keeps only weather rows that matched to a labeled location
                    df = df.dropna(subset=["location_id"])

                    logger.info(
                        f"Matched {len(df)} weather records to weevil locations"
                    )
                else:
                    logger.warning(
                        "No spatial_coverage path in config, skipping location_id matching"
                    )

        logger.info("Weather data processing completed")
        return df

    def match_locations_to_place_id(
        self, location_df: pd.DataFrame, spatial_coverage: pd.DataFrame
    ) -> gpd.GeoDataFrame:
        """
        Match locations to `place_id` in spatial_coverage based on latitude and longitude.
        `place_id` is the identifier for a specific grid cell in the spatial coverage data.

        Parameters:
            location_df (pd.DataFrame): DataFrame with location_id, 'latitude', and 'longitude' columns.
            spatial_coverage (pd.DataFrame): DataFrame with 'place_id' and 'boundary' columns (WKT format).

        Returns:
            gpd.GeoDataFrame: Original location_df with additional 'place_id' column.
        """
        # Turn to geodataframe
        location_gdf = gpd.GeoDataFrame(
            location_df,
            geometry=gpd.points_from_xy(location_df.longitude, location_df.latitude),
            crs="EPSG:4326",
        )
        # Convert boundary column from WKT string to shapely geometry
        spatial_coverage["boundary"] = spatial_coverage["boundary"].apply(wkt.loads)
        spatial_coverage_gdf = gpd.GeoDataFrame(
            spatial_coverage, geometry="boundary", crs="EPSG:4326"
        )
        # Merge with spatial coverage to get place_id and boundary
        matched_gdf = gpd.sjoin(
            location_gdf,
            spatial_coverage_gdf[["place_id", "boundary"]],
            how="left",
            predicate="within",
        )
        matched_gdf.drop(columns=["index_right"], inplace=True)
        return matched_gdf

    @staticmethod
    def _build_post_event_training_rows(
        weather_data: pd.DataFrame,
        weevil_data: pd.DataFrame,
        horizon_days: int = 5,
    ) -> pd.DataFrame:
        """
        Build short-horizon post-event rows using real weather rows.

        Output rows are aligned on the same training schema used after the
        weather/weevil forward merge: one row per available post-event weather day,
        with event metadata attached. The legacy non-negative target stays at zero,
        while the signed target keeps counting down after the event date.
        """
        if horizon_days <= 0:
            return pd.DataFrame()

        if "stage_date" not in weevil_data.columns:
            if weevil_data.index.name == "stage_date":
                weevil_data = weevil_data.reset_index()
            elif isinstance(weevil_data.index, pd.DatetimeIndex):
                weevil_data = weevil_data.reset_index().rename(
                    columns={weevil_data.index.name or "index": "stage_date"}
                )

        required_weather = {"date", "location_id", "pest_year"}
        required_weevil = {
            "stage_date",
            "location_id",
            "pest_year",
            "stage_id",
            "day_of_year",
        }
        missing_weather = required_weather - set(weather_data.columns)
        missing_weevil = required_weevil - set(weevil_data.columns)
        if missing_weather:
            raise ValueError(
                f"weather_data missing required columns for post-event rows: {sorted(missing_weather)}"
            )
        if missing_weevil:
            raise ValueError(
                f"weevil_data missing required columns for post-event rows: {sorted(missing_weevil)}"
            )

        weather = weather_data.copy()
        weather["date"] = pd.to_datetime(weather["date"])
        weather["location_id"] = weather["location_id"].astype(str)
        weather["pest_year"] = pd.to_numeric(weather["pest_year"], errors="coerce")

        events = weevil_data.copy()
        events["stage_date"] = pd.to_datetime(events["stage_date"])
        events["location_id"] = events["location_id"].astype(str)
        events["pest_year"] = pd.to_numeric(events["pest_year"], errors="coerce")
        events["stage_id"] = pd.to_numeric(events["stage_id"], errors="coerce").astype(
            int
        )

        rows: List[pd.DataFrame] = []
        for event in events.itertuples(index=False):
            stage_date = pd.to_datetime(event.stage_date)
            end_date = stage_date + pd.Timedelta(days=horizon_days)

            sub = weather[
                (weather["location_id"] == str(event.location_id))
                & (weather["pest_year"] == event.pest_year)
                & (weather["date"] >= stage_date)
                & (weather["date"] <= end_date)
            ].copy()

            if sub.empty:
                continue

            sub["days_since_event"] = (sub["date"] - stage_date).dt.days.astype(int)
            sub["stage_date"] = stage_date
            sub["day_of_year"] = int(event.day_of_year)
            sub["stage_id"] = int(event.stage_id)
            # Legacy non-negative target flow kept for rollback/reference.
            sub["days_to_event"] = 0
            sub["signed_days_to_event"] = -sub["days_since_event"]

            if "latitude" not in sub.columns or sub["latitude"].isna().all():
                sub["latitude"] = getattr(event, "latitude", np.nan)
            if "longitude" not in sub.columns or sub["longitude"].isna().all():
                sub["longitude"] = getattr(event, "longitude", np.nan)

            rows.append(sub)

        if not rows:
            return pd.DataFrame()

        out = pd.concat(rows, ignore_index=True)
        out = out.drop_duplicates(
            subset=["date", "location_id", "pest_year", "stage_id"],
            keep="first",
        ).copy()
        return out

    def _augment_training_data_with_post_event_zero_rows(
        self,
        model_data: pd.DataFrame,
        weather_data: pd.DataFrame,
        weevil_data: pd.DataFrame,
        horizon_days: int = 5,
    ) -> pd.DataFrame:
        """
        Add post-event rows labeled with zero remaining days using real feature rows.
        """
        post_event_rows = self._build_post_event_training_rows(
            weather_data=weather_data,
            weevil_data=weevil_data,
            horizon_days=horizon_days,
        )
        if post_event_rows.empty:
            return model_data

        base = model_data.copy().reset_index().rename(columns={"index": "date"})
        if "date" not in base.columns:
            raise ValueError("model_data must carry the date index for augmentation")

        cols = list(
            dict.fromkeys(base.columns.tolist() + post_event_rows.columns.tolist())
        )
        base = base.reindex(columns=cols)
        post_event_rows = post_event_rows.reindex(columns=cols)

        augmented = pd.concat([base, post_event_rows], ignore_index=True)
        augmented["date"] = pd.to_datetime(augmented["date"])
        augmented["location_id"] = augmented["location_id"].astype(str)
        if "stage_id" in augmented.columns:
            augmented["stage_id"] = pd.to_numeric(
                augmented["stage_id"], errors="coerce"
            )
        if "pest_year" in augmented.columns:
            augmented["pest_year"] = pd.to_numeric(
                augmented["pest_year"], errors="coerce"
            )

        augmented = augmented.sort_values(
            ["location_id", "pest_year", "stage_id", "date"]
        ).drop_duplicates(
            subset=["date", "location_id", "pest_year", "stage_id"],
            keep="last",
        )
        if "days_to_event" in augmented.columns:
            augmented["days_to_event"] = pd.to_numeric(
                augmented["days_to_event"], errors="coerce"
            ).clip(lower=0)
        if "signed_days_to_event" in augmented.columns:
            augmented["signed_days_to_event"] = pd.to_numeric(
                augmented["signed_days_to_event"], errors="coerce"
            )
        augmented = augmented.set_index("date").sort_index()
        return augmented

    def prepare_training_data(
        self,
        weevil_data: pd.DataFrame,
        weather_data: pd.DataFrame,
        test_date: str,
        features: List[str],
        target: str,
        window_years: Optional[int] = None,
        post_event_training_days: Optional[int] = None,
        post_event_zero_days: Optional[int] = None,
        include_test_date: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training data by merging weevil and weather data.

        Supports either:
        - Legacy behavior (default): keep data from >= 2018-03-01
        - Fixed rolling window: keep data within [test_date - window_years, test_date)
          or [test_date - window_years, test_date] when ``include_test_date=True``

        Args:
            weevil_data: Processed weevil data
            weather_data: Processed weather data
            test_date: Date to split training/test data (anchor for rolling window)
            features: List of feature column names
            target: Target column name
            window_years: If provided, use a fixed rolling window of N years prior to test_date.
            include_test_date: Whether rows dated exactly ``test_date`` are retained.

        Returns:
            Tuple of (X_train, y_train) DataFrames
        """
        logger.info("Preparing training data...")

        test_ts = pd.to_datetime(test_date)

        if window_years is not None:
            if window_years <= 0:
                raise ValueError(
                    f"window_years must be a positive integer, got {window_years}"
                )
            window_start = test_ts - pd.DateOffset(years=window_years)
            boundary = "]" if include_test_date else ")"
            logger.info(
                f"Using fixed rolling window: [{window_start.date()}, {test_ts.date()}{boundary}"
            )
        else:
            window_start = pd.to_datetime("2018-03-01")
            logger.info(f"Using legacy cutoff: >= {window_start.date()}")

        weevil_data = weevil_data.set_index("stage_date").rename(
            columns={"year": "pest_year"}
        )
        dfs = []

        # Merge weather data with weevil data for each stage
        weather_sorted = weather_data.set_index("date").sort_index()

        for stage in range(1, 6):
            stage_data = weevil_data[weevil_data.stage_id == stage].sort_index()

            temp = pd.merge_asof(
                weather_sorted,
                stage_data,
                left_index=True,
                right_index=True,
                direction="forward",
                by=["location_id", "pest_year"],
                suffixes=("_weather", "_weevil"),
            )

            # Unify lat/lon columns
            if "latitude_weevil" in temp.columns:
                temp["latitude"] = temp["latitude_weevil"]
                temp.drop(columns=["latitude_weather"], errors="ignore", inplace=True)
            elif "latitude" not in temp.columns and "latitude_weather" in temp.columns:
                temp["latitude"] = temp["latitude_weather"]

            if "longitude_weevil" in temp.columns:
                temp["longitude"] = temp["longitude_weevil"]
                temp.drop(columns=["longitude_weather"], errors="ignore", inplace=True)
            elif (
                "longitude" not in temp.columns and "longitude_weather" in temp.columns
            ):
                temp["longitude"] = temp["longitude_weather"]

            dfs.append(temp)

        model_data = pd.concat(dfs)

        # Apply time filtering
        if include_test_date:
            model_data = model_data[
                (model_data.index >= window_start) & (model_data.index <= test_ts)
            ]
        else:
            model_data = model_data[
                (model_data.index >= window_start) & (model_data.index < test_ts)
            ]

        model_data.dropna(subset=["day_of_year"], inplace=True)

        # Legacy non-negative target flow kept for rollback/reference:
        # model_data["days_to_event"] = model_data["day_of_year"] - model_data.index.dayofyear
        target_delta = model_data["day_of_year"] - model_data.index.dayofyear
        model_data["days_to_event"] = target_delta
        model_data["signed_days_to_event"] = target_delta

        if post_event_training_days is None:
            post_event_training_days = post_event_zero_days
        if post_event_training_days is None:
            post_event_training_days = 5

        horizon_days = int(post_event_training_days)
        model_data = self._augment_training_data_with_post_event_zero_rows(
            model_data=model_data,
            weather_data=weather_data,
            weevil_data=weevil_data,
            horizon_days=horizon_days,
        )
        model_data["days_to_event"] = pd.to_numeric(
            model_data["days_to_event"], errors="coerce"
        ).clip(lower=0)
        model_data["signed_days_to_event"] = pd.to_numeric(
            model_data["signed_days_to_event"], errors="coerce"
        )
        model_data.sort_index(inplace=True)

        # Train split (after windowing this is effectively all remaining rows)
        if include_test_date:
            train_data = model_data[model_data.index <= test_ts]
        else:
            train_data = model_data[model_data.index < test_ts]
        train_data = train_data.dropna(subset=[target])
        train_data = train_data.dropna(subset=features)

        x_train = train_data[features]
        y_train = train_data[[target]]

        if len(x_train) != len(y_train):
            raise ValueError(
                f"Training data mismatch: x_train has {len(x_train)} rows, "
                f"y_train has {len(y_train)} rows"
            )

        if window_years is not None:
            logger.info(
                f"Training data prepared (window_years={window_years}): {len(x_train)} samples "
                f"from {window_start.date()} to "
                f"{test_ts.date() if include_test_date else (test_ts - pd.Timedelta(days=1)).date()} "
                f"(inclusive)"
            )
        else:
            logger.info(f"Training data prepared: {len(x_train)} samples")

        return x_train, y_train

    # When the test data includes historical records for model evaluation.
    def prepare_test_data(
        self,
        weather_data: pd.DataFrame,
        features: List[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Prepare test data for prediction.

        - Avoids .loc slicing on a non-monotonic DatetimeIndex by using boolean filtering.
        - Keeps output indexed by date (DatetimeIndex) for downstream compatibility.
        """

        if end_date is None:
            end_date = start_date

        logger.info(f"Preparing test data from {start_date} to {end_date}")

        # --- Robust date filtering (does NOT require monotonic index) ---
        wd = weather_data.copy()
        wd["date"] = pd.to_datetime(wd["date"])

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        raw_test = wd[(wd["date"] >= start_dt) & (wd["date"] <= end_dt)].copy()

        # If no weather rows exist for the date(s), return empty
        if raw_test.empty:
            logger.warning(f"No weather data found between {start_date} and {end_date}")
            return pd.DataFrame()

        # Stable ordering (useful when end_date > start_date)
        raw_test = raw_test.sort_values("date")

        # Set index back to date (same behavior as original)
        raw_test = raw_test.set_index("date")

        stage_ids = tuple(
            model_stage_ids_from_config(
                self.db_manager.config.config
                if self.db_manager.config is not None
                else None
            )
        )
        # --- Construct stage-expanded test set ---
        x_test = pd.concat([raw_test] * len(stage_ids), axis=0)

        # stage_id is not in weather_data; we add it here
        x_test = x_test.assign(stage_id=np.repeat(stage_ids, len(raw_test)))

        # Keep only required feature columns for model input
        columns_to_keep = features.copy()

        # Preserve identifiers for filtering/output if present.
        for col in ["location_id", "place_id"]:
            if col in x_test.columns and col not in columns_to_keep:
                columns_to_keep.append(col)

        # Also preserve lat/lon if present and you need them for output (optional)
        # If your downstream expects these columns but they are not in `features`,
        # uncomment the following lines:
        # for col in ["latitude", "longitude"]:
        #     if col in x_test.columns and col not in columns_to_keep:
        #         columns_to_keep.append(col)

        x_test = x_test[columns_to_keep]

        return x_test

    # only for test(ie.There doesn't exist historical weevil information for the test data)
    def prepare_test_data_latest(
        self,
        weather_data: pd.DataFrame,
        features: List[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Prepare test data for prediction.

        - Robust date filtering via boolean mask.
        - Output indexed by date (DatetimeIndex).
        - stage_id is created here.
        - Keeps identifier columns needed for downstream output/debugging.
        """
        if end_date is None:
            end_date = start_date

        logger.info(f"Preparing test data from {start_date} to {end_date}")

        wd = weather_data.copy()
        wd["date"] = pd.to_datetime(wd["date"])

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        raw_test = wd[(wd["date"] >= start_dt) & (wd["date"] <= end_dt)].copy()
        if raw_test.empty:
            logger.warning(f"No weather data found between {start_date} and {end_date}")
            return pd.DataFrame()

        raw_test = raw_test.sort_values("date").set_index("date")

        stage_ids = tuple(
            model_stage_ids_from_config(
                self.db_manager.config.config
                if self.db_manager.config is not None
                else None
            )
        )
        # --- stage-expanded test set ---
        x_test = pd.concat([raw_test] * len(stage_ids), axis=0)
        x_test = x_test.assign(stage_id=np.repeat(stage_ids, len(raw_test)))

        # Keep model features + output/debug columns if present
        columns_to_keep = list(features)
        for col in ["latitude", "longitude", "place_id", "location_id"]:
            if col in x_test.columns and col not in columns_to_keep:
                columns_to_keep.append(col)

        x_test = x_test[columns_to_keep]
        return x_test

    # Helper methods
    def _convert_celsius_to_fahrenheit(self, celsius: pd.Series) -> pd.Series:
        """Convert Celsius to Fahrenheit

        Args:
            celsius: Series containing Celsius temperature

        Returns:
            Series with Fahrenheit temperature
        """
        return (celsius * 9 / 5) + 32

    def _calc_growing_degree_days(
        self, t_min: pd.Series, t_max: pd.Series, base_temp: float
    ) -> pd.Series:
        """
        Calculate Growing Degree Days

        Args:
            t_min: Series containing minimum temperature
            t_max: Series containing maximum temperature
            base_temp: Base temperature for growing degree days

        Returns:
            Series with growing degree days
        """
        return ((t_max + t_min) / 2 - base_temp).clip(lower=0)

    def _calc_annual_cumulative(
        self,
        df: pd.DataFrame,
        start_month: int,
        start_day: int,
        target_cols: Optional[List[str]] = None,
        group_col: str = "place_id",
    ) -> pd.DataFrame:
        """
        Calculate annual cumulative values

        Args:
            df: DataFrame containing weather data
            start_month: Start month
            start_day: Start day
            target_cols: List of target columns to calculate annual cumulative values
            group_col: Column to group by

        Returns:
            DataFrame with annual cumulative values
        """
        if target_cols is None:
            candidate_cols = [
                "gdd_air",
                "precip_total_mm",
                "humidity_mean_pct",
                "cdd_air",
            ]
            target_cols = [col for col in candidate_cols if col in df.columns]

        df["pest_year"] = df.date.apply(
            lambda x: self._get_pest_year(x, start_month, start_day)
        )

        grouped = df.groupby(["pest_year", group_col])
        for col in target_cols:
            df["cumu_" + col] = grouped[col].cumsum()

        return df

    def _get_pest_year(self, date_obj, start_month: int, start_day: int) -> int:
        """
        Determine pest year based on start date

        Args:
            date_obj: Date object
            start_month: Start month
            start_day: Start day

        Returns:
            Pest year
        """
        return (
            date_obj.year - 1
            if (
                date_obj.month < start_month
                or (date_obj.month == start_month and date_obj.day < start_day)
            )
            else date_obj.year
        )

    def _calc_rolling_avg(self, df: pd.DataFrame, windows: int = 7, target_cols=None):
        """Compute rolling features within (pest_year, place_id) groups."""
        if target_cols is None:
            candidate_cols = [
                "gdd_air",
                "precip_total_mm",
                "humidity_mean_pct",
                "cdd_air",
            ]
            target_cols = [col for col in candidate_cols if col in df.columns]

        group_cols = ["pest_year", "place_id"]
        for col in ["date"] + group_cols:
            if col not in df.columns:
                raise KeyError(
                    f"Required column '{col}' not found for grouped rolling."
                )

        df = df.sort_values(group_cols + ["date"]).copy()

        for col in target_cols:
            roll_name = "rolling_" + col
            df[roll_name] = (
                df.groupby(group_cols, sort=False)[col]
                .rolling(window=windows, min_periods=1)
                .mean()
                .reset_index(level=group_cols, drop=True)
            )
        return df


if __name__ == "__main__":
    config = WeevillTrakConfig("app/config/weeviltrak_v2.3.yml")
    bucket_name = config.config.get("data_source", {}).get("s3_bucket", "sps-ds-bucket")
    data_processing_service = DataPreparationService(
        DatabaseManager(config), S3Manager(bucket_name)
    )
    weevil_data = data_processing_service.pull_weevil_data(today="2025-11-11")

    print(weevil_data)
    print("weevil_data columns: ", weevil_data.columns)
    print("weevil_data['stage_id']: ", weevil_data["stage_id"])

    # weather_data = data_processing_service.query_weather_data_for_weeviltrak_locs(config, end_date="2025-11-11")
    # print(weather_data)

    # processed_weather_data = data_processing_service.process_weather_data(weather_data, config, weevil_data)
    # print(processed_weather_data['latitude'])
    # print(processed_weather_data['longitude'])
    # print(processed_weather_data['location_id'])
    # print("processed_weather_data columns: ", processed_weather_data.columns)

    # x_train, y_train = data_processing_service.prepare_training_data(weevil_data, processed_weather_data, test_date="2025-11-11", features=["air_temp_max_c", "air_temp_min_c", "precip_total_mm", "humidity_mean_pct", "soil_temp_c", "soil_moisture_mean_m3m3", "sun_duration_min"], target="days_to_event")
    # print(x_train)
    # print(y_train)
    # print("x_train columns: ", x_train.columns)
    # print("y_train columns: ", y_train.columns)

    # x_test = data_processing_service.prepare_test_data(processed_weather_data, features=["air_temp_max_c", "air_temp_min_c", "precip_total_mm", "humidity_mean_pct", "soil_temp_c", "soil_moisture_mean_m3m3", "sun_duration_min"], start_date="2025-11-11", end_date="2025-11-11")
    # print("x_test columns: ", x_test.columns)
    # print("x_test: ", x_test)

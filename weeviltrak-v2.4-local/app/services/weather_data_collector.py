"""
Weather data collection service for pulling weather data from Redshift.
"""
import os
import pandas as pd
import multiprocessing as mp
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from tqdm import tqdm

from app.services.database_service import DatabaseManager
from app.settings import setup_logger
from app.services.s3_service import S3Manager

logger = setup_logger()

# Global connection for multiprocessing
conn = None
dbm = None

def _init_conn():
    """Initialize database connection for worker processes"""
    # global conn
    global conn, dbm
    # db_manager = DatabaseManager()
    # conn = db_manager.get_connection()
    dbm = DatabaseManager()
    conn = dbm.get_connection()




def build_query_10by10_all_features(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> str:
    """Build 10by10 query with all features (daily nearest grid point)."""
    return f"""
        SELECT
            place_id,
            lat AS centroid_lat,
            lon AS centroid_lon,
            date,
            timeresolution,
            weather_data_domains,

            -- Weather features
            daylight_duration_min_surface_sum AS sun_duration_min,

            dewpoint_temperature_c_2_m_above_gnd_avg,
            leaf_wetness_probability_pct_2_m_above_gnd_avg,

            precipitation_total_grid_mm_surface_sum,
            precipitation_total_mm_surface_sum,
            precipitation_total_sat_mm_surface_sum,

            reference_evapotranspiration_mm_2_m_above_gnd_sum,
            shortwave_radiation_w_per_m2_surface_sum,

            relative_humidity_pct_2_m_above_gnd_avg,
            relative_humidity_pct_2_m_above_gnd_max,
            relative_humidity_pct_2_m_above_gnd_min,

            vapor_pressure_deficit_hpa_2_m_above_gnd_max,

            wind_gust_km_per_h_surface_max,
            wind_speed_km_per_h_2_m_above_gnd_avg,
            wind_speed_km_per_h_2_m_above_gnd_max,

            temperature_c_2_m_above_gnd_avg,
            temperature_c_2_m_above_gnd_max,
            temperature_c_2_m_above_gnd_min,
            temperature_c_2_m_above_gnd_sum,

            temperature_c_surface_avg,

            temperature_c_0to10cm_below_gnd_avg,
            temperature_c_10to30cm_below_gnd_avg,
            temperature_c_30to100cm_below_gnd_avg,

            soil_water_content_m3_per_m3_0to10cm_below_gnd_avg,
            soil_water_content_m3_per_m3_10to30cm_below_gnd_avg,
            soil_water_content_m3_per_m3_30to100cm_below_gnd_avg,

            soil_transpirable_water_fraction_0to100cm_below_gnd_avg,

            distance_between_sample_and_nearest_point_in_km
        FROM (
            SELECT
            place_id,
            lat, lon, date,
            timeresolution,
            weather_data_domains,

            daylight_duration_min_surface_sum,

            dewpoint_temperature_c_2_m_above_gnd_avg,
            leaf_wetness_probability_pct_2_m_above_gnd_avg,

            precipitation_total_grid_mm_surface_sum,
            precipitation_total_mm_surface_sum,
            precipitation_total_sat_mm_surface_sum,

            reference_evapotranspiration_mm_2_m_above_gnd_sum,
            shortwave_radiation_w_per_m2_surface_sum,

            relative_humidity_pct_2_m_above_gnd_avg,
            relative_humidity_pct_2_m_above_gnd_max,
            relative_humidity_pct_2_m_above_gnd_min,

            vapor_pressure_deficit_hpa_2_m_above_gnd_max,

            wind_gust_km_per_h_surface_max,
            wind_speed_km_per_h_2_m_above_gnd_avg,
            wind_speed_km_per_h_2_m_above_gnd_max,

            temperature_c_2_m_above_gnd_avg,
            temperature_c_2_m_above_gnd_max,
            temperature_c_2_m_above_gnd_min,
            temperature_c_2_m_above_gnd_sum,

            temperature_c_surface_avg,

            temperature_c_0to10cm_below_gnd_avg,
            temperature_c_10to30cm_below_gnd_avg,
            temperature_c_30to100cm_below_gnd_avg,

            soil_water_content_m3_per_m3_0to10cm_below_gnd_avg,
            soil_water_content_m3_per_m3_10to30cm_below_gnd_avg,
            soil_water_content_m3_per_m3_30to100cm_below_gnd_avg,

            soil_transpirable_water_fraction_0to100cm_below_gnd_avg,

            ST_DistanceSphere(
                ST_Point(lon, lat),
                ST_Point({longitude}, {latitude})
            ) / 1000 AS distance_between_sample_and_nearest_point_in_km,

            ROW_NUMBER() OVER (
                PARTITION BY date
                ORDER BY ST_DistanceSphere(
                    ST_Point(lon, lat),
                    ST_Point({longitude}, {latitude})
                )
            ) AS row_num
        FROM spectrum_schema.mio171_weather_10x10_pivoted_archive
        WHERE lon BETWEEN {longitude} - 0.5 AND {longitude} + 0.5
            AND lat BETWEEN {latitude} - 0.5 AND {latitude} + 0.5
            AND date BETWEEN '{start_date}' AND '{end_date}'
    ) subquery
    WHERE row_num = 1;
"""

def _query_single_location_task(args: Tuple[str, float, float, str, str, str]) -> pd.DataFrame:
    """
    Worker task: (loc, latitude, longitude, start_date, end_date, resolution)
    """
    loc, latitude, longitude, start_date, end_date, resolution = args

    if resolution != "10by10":
        # Keep simple: only 10by10 is implemented for locations pull here
        return pd.DataFrame()

    query = build_query_10by10_all_features(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
    )

    try:
        # global conn
        global conn, dbm
        if conn is None:
            if dbm is None:
                dbm = DatabaseManager()
            conn = dbm.get_connection()
        # Optional: handle dropped connections
        try:
            # redshift_connector has is_closed() in many versions
            if hasattr(conn, "is_closed") and conn.is_closed():
                conn = dbm.get_connection()
        except Exception:
            # If status check fails, try reconnect once
            conn = dbm.get_connection()

        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(result, columns=columns)

        if df.empty:
            return df

        # Add location labels
        df["loc"] = str(loc)
        df["latitude"] = float(latitude)
        df["longitude"] = float(longitude)
        return df

    except Exception as e:
        logger.error(f"Error querying location task {args}: {e}")
        return pd.DataFrame()


class WeatherDataCollector:
    """
    Service for collecting weather data from Redshift database.
    Supports both single location and parallel data collection.
    """

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize weather data collector.

        Args:
            db_manager: Database manager for accessing Redshift
        """
        self.db_manager = db_manager
        logger.info("WeatherDataCollector initialized")

    def pull_single_location(
        self,
        location_id: int,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        resolution: str = '10by10'
    ) -> pd.DataFrame:
        """
        Pull weather data for a single location from Redshift.

        Args:
            location_id: Location identifier
            latitude: Latitude of the location
            longitude: Longitude of the location
            start_date: Start date in 'YYYYMMDD' format
            end_date: End date in 'YYYYMMDD' format
            resolution: Weather data resolution ('10by10' or '30by30')

        Returns:
            DataFrame with weather data for the location
        """
        logger.info(
            f"Pulling {resolution} weather data for location {location_id} "
            f"at ({latitude}, {longitude}) from {start_date} to {end_date}"
        )

        query = self._build_query(latitude, longitude, start_date, end_date, resolution)

        with self.db_manager as db:
            connection = db.get_connection()
            try:
                cursor = connection.cursor()
                cursor.execute(query)
                result = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                df = pd.DataFrame(result, columns=columns)
                df = df.assign(
                    latitude=latitude,
                    longitude=longitude,
                    location_id=location_id
                )
                return df
            except Exception as e:
                logger.error(f"Error pulling weather data for location {location_id}: {e}")
                return pd.DataFrame()  # Return empty DataFrame on error

    def pull_parallel(
        self,
        place_ids: List[str],
        start_date: str,
        end_date: str,
        chunk_months: int = 2
    ) -> pd.DataFrame:
        """
        Pull weather data for multiple locations in parallel.

        Args:
            place_ids: List of place IDs to pull data for
            start_date: Start date in 'YYYYMMDD' format
            end_date: End date in 'YYYYMMDD' format
            chunk_months: Number of months per chunk (default: 2)

        Returns:
            Combined DataFrame with weather data for all locations
        """
        logger.info(
            f"Pulling weather data for {len(place_ids)} places "
            f"from {start_date} to {end_date}"
        )

        # Generate date chunks
        chunks = self._generate_date_chunks(start_date, end_date, chunk_months)

        # Create tasks
        tasks = [
            (pid, chunk_start, chunk_end)
            for pid in place_ids
            for chunk_start, chunk_end in chunks
        ]

        logger.info(f"Total tasks to process: {len(tasks)}")

        # Execute in parallel
        with mp.Pool(processes=mp.cpu_count() - 1, initializer=_init_conn) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(self._query_single_task, tasks),
                    total=len(tasks)
                )
            )

        # Filter out empty results and combine
        valid_results = [r for r in results if not r.empty]
        if not valid_results:
            logger.warning("No valid results from parallel pull")
            return pd.DataFrame()

        df = pd.concat(valid_results, ignore_index=True)
        logger.info(f"Successfully pulled {len(df)} weather records")
        return df

    def _build_query(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        resolution: str
    ) -> str:
        """Build SQL query for weather data retrieval"""
        if resolution == '10by10':
            # return f"""
            #     SELECT
            #         lat AS centroid_lat,
            #         lon AS centroid_lon,
            #         date,
            #         temperature_c_2_m_above_gnd_max AS air_temp_max_c,
            #         temperature_c_2_m_above_gnd_min AS air_temp_min_c,
            #         temperature_c_2_m_above_gnd_avg AS air_temp_avg_c,
            #         temperature_c_0to10cm_below_gnd_avg AS soil_temp_c,
            #         precipitation_total_grid_mm_surface_sum AS precip_total_mm,
            #         relative_humidity_pct_2_m_above_gnd_avg AS humidity_mean_pct,
            #         soil_water_content_m3_per_m3_0to10cm_below_gnd_avg AS soil_moisture_mean_m3m3,
            #         daylight_duration_min_surface_sum AS sun_duration_min,
            #         distance_between_sample_and_nearest_point_in_km
            #     FROM (
            #         SELECT
            #             lat, lon, date,
            #             temperature_c_2_m_above_gnd_max,
            #             temperature_c_2_m_above_gnd_min,
            #             temperature_c_2_m_above_gnd_avg,
            #             temperature_c_0to10cm_below_gnd_avg,
            #             precipitation_total_grid_mm_surface_sum,
            #             relative_humidity_pct_2_m_above_gnd_avg,
            #             soil_water_content_m3_per_m3_0to10cm_below_gnd_avg,
            #             daylight_duration_min_surface_sum,
            #             ST_DistanceSphere(
            #                 ST_Point(lon, lat),
            #                 ST_Point({longitude}, {latitude})
            #             ) / 1000 AS distance_between_sample_and_nearest_point_in_km,
            #             ROW_NUMBER() OVER (
            #                 PARTITION BY date
            #                 ORDER BY ST_DistanceSphere(
            #                     ST_Point(lon, lat),
            #                     ST_Point({longitude}, {latitude})
            #                 )
            #             ) AS row_num
            #         FROM spectrum_schema.mio171_weather_10x10_pivoted_archive
            #         WHERE lon BETWEEN {longitude} - 0.5 AND {longitude} + 0.5
            #             AND lat BETWEEN {latitude} - 0.5 AND {latitude} + 0.5
            #             AND date BETWEEN '{start_date}' AND '{end_date}'
            #     ) subquery
            #     WHERE row_num = 1;
            # """
            return build_query_10by10_all_features(
                latitude=latitude,
                longitude=longitude,
                start_date=start_date,
                end_date=end_date,
            )

        elif resolution == '30by30':
            return f"""
                SELECT
                    lat AS matched_lat,
                    lon AS matched_lon,
                    weather_date AS date,
                    temperature_c_2_m_above_gnd_max AS air_temp_max_c,
                    temperature_c_2_m_above_gnd_min AS air_temp_min_c,
                    soil_temperature_c_0_10_cm_down_max AS soil_temp_max_c,
                    soil_temperature_c_0_10_cm_down_min AS soil_temp_min_c,
                    precipitation_total_mm_surface_sum AS precip_total_mm,
                    relative_humidity_pct_2_m_above_gnd_avg AS humidity_mean_pct,
                    soil_moisture_m3_per_m3_0_10_cm_down_avg AS soil_moisture_mean_m3m3,
                    sunshine_duration_min_surface_sum AS sun_duration_min,
                    distance_between_sample_and_nearest_point_in_km
                FROM (
                    SELECT
                        lat, lon, weather_date,
                        temperature_c_2_m_above_gnd_max,
                        temperature_c_2_m_above_gnd_min,
                        soil_temperature_c_0_10_cm_down_max,
                        soil_temperature_c_0_10_cm_down_min,
                        precipitation_total_mm_surface_sum,
                        relative_humidity_pct_2_m_above_gnd_avg,
                        soil_moisture_m3_per_m3_0_10_cm_down_avg,
                        sunshine_duration_min_surface_sum,
                        ST_DistanceSphere(
                            ST_Point(lon, lat),
                            ST_Point({longitude}, {latitude})
                        ) / 1000 AS distance_between_sample_and_nearest_point_in_km,
                        ROW_NUMBER() OVER (
                            PARTITION BY weather_date
                            ORDER BY ST_DistanceSphere(
                                ST_Point(lon, lat),
                                ST_Point({longitude}, {latitude})
                            )
                        ) AS row_num
                    FROM mio171_t_weather_30x30_fact
                    WHERE lon BETWEEN {longitude} - 0.5 AND {longitude} + 0.5
                        AND lat BETWEEN {latitude} - 0.5 AND {latitude} + 0.5
                        AND weather_date BETWEEN '{start_date}' AND '{end_date}'
                ) subquery
                WHERE row_num = 1;
            """
        else:
            raise ValueError("Invalid resolution. Choose '10by10' or '30by30'.")


    def pull_locations_parallel_full_rebuild_to_s3(
        self,
        locations_path: str,
        resolution: str = "10by10",
        chunk_months: int = 2,
        processes: Optional[int] = None,
        s3_bucket: str = "sps-ds-bucket",
        s3_key: str = "weeviltrak_data/historical_weather_data_10by10_updated_locations.h5",
    ) -> pd.DataFrame:
        """
        Pull weather for each row in locations file in parallel and upload to S3.
        Dedupe by loc + date.
        """
        logger.info(f"Reading locations file: {locations_path}")

        loc_df = self._read_locations_file(locations_path)
        tasks: List[Tuple[str, float, float, str, str, str]] = []

        for _, r in loc_df.iterrows():
            loc = str(r["loc"])
            lat = float(r["latitude"])
            lon = float(r["longitude"])
            start_yyyymmdd = r["start_yyyymmdd"]
            end_yyyymmdd = r["end_yyyymmdd"]

            chunks = self._generate_date_chunks(start_yyyymmdd, end_yyyymmdd, chunk_months)
            for cs, ce in chunks:
                tasks.append((loc, lat, lon, cs, ce, resolution))

        logger.info(f"Total location tasks: {len(tasks)}")

        procs = processes if processes is not None else max(1, mp.cpu_count() - 1)
        logger.info(f"Using processes: {procs}")

        # with mp.Pool(processes=procs, initializer=_init_conn) as pool:
        #     results = list(
        #         tqdm(
        #             pool.imap_unordered(_query_single_location_task, tasks),
        #             total=len(tasks),
        #         )
        #     )
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=procs, initializer=_init_conn) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_query_single_location_task, tasks),
                    total=len(tasks),
                )
            )


        valid = [d for d in results if d is not None and not d.empty]
        if not valid:
            logger.warning("No valid results from locations pull")
            df_all = pd.DataFrame()
        else:
            df_all = pd.concat(valid, ignore_index=True)

        if df_all.empty:
            logger.warning("Final DataFrame is empty. Uploading empty file to S3.")
            s3 = S3Manager(bucket_name=s3_bucket)
            s3.upload_file(df_all, s3_key)
            return df_all

        # Keep date as string
        df_all["date"] = df_all["date"].astype(str)

        # Dedupe by loc + date
        df_all.sort_values(["loc", "date"], inplace=True)
        df_all.drop_duplicates(subset=["loc", "date"], keep="last", inplace=True)

        # Optional numeric casting for common columns
        numeric_cols = [
            "centroid_lat", "centroid_lon",
            "sun_duration_min",
            "dewpoint_temperature_c_2_m_above_gnd_avg",
            "leaf_wetness_probability_pct_2_m_above_gnd_avg",
            "precipitation_total_grid_mm_surface_sum",
            "precipitation_total_mm_surface_sum",
            "precipitation_total_sat_mm_surface_sum",
            "reference_evapotranspiration_mm_2_m_above_gnd_sum",
            "shortwave_radiation_w_per_m2_surface_sum",
            "relative_humidity_pct_2_m_above_gnd_avg",
            "relative_humidity_pct_2_m_above_gnd_max",
            "relative_humidity_pct_2_m_above_gnd_min",
            "vapor_pressure_deficit_hpa_2_m_above_gnd_max",
            "wind_gust_km_per_h_surface_max",
            "wind_speed_km_per_h_2_m_above_gnd_avg",
            "wind_speed_km_per_h_2_m_above_gnd_max",
            "temperature_c_2_m_above_gnd_avg",
            "temperature_c_2_m_above_gnd_max",
            "temperature_c_2_m_above_gnd_min",
            "temperature_c_2_m_above_gnd_sum",
            "temperature_c_surface_avg",
            "temperature_c_0to10cm_below_gnd_avg",
            "temperature_c_10to30cm_below_gnd_avg",
            "temperature_c_30to100cm_below_gnd_avg",
            "soil_water_content_m3_per_m3_0to10cm_below_gnd_avg",
            "soil_water_content_m3_per_m3_10to30cm_below_gnd_avg",
            "soil_water_content_m3_per_m3_30to100cm_below_gnd_avg",
            "soil_transpirable_water_fraction_0to100cm_below_gnd_avg",
            "distance_between_sample_and_nearest_point_in_km",
            "latitude", "longitude",
        ]
        for col in numeric_cols:
            if col in df_all.columns:
                df_all[col] = pd.to_numeric(df_all[col], errors="coerce").astype("float64")

        # Upload to S3
        logger.info(f"Uploading full rebuild to s3://{s3_bucket}/{s3_key}")
        s3 = S3Manager(bucket_name=s3_bucket)
        s3.upload_file(df_all, s3_key)
        logger.info(f"Upload done. Rows: {len(df_all)}")

        return df_all


    def _generate_date_chunks(
        self,
        start: str,
        end: str,
        chunk_months: int = 2
    ) -> List[Tuple[str, str]]:
        """
        Split date range into chunks to avoid Redshift query timeouts.

        Args:
            start: Start date in 'YYYYMMDD' format
            end: End date in 'YYYYMMDD' format
            chunk_months: Number of months per chunk

        Returns:
            List of (start_date, end_date) tuples
        """
        start_dt = datetime.strptime(start, "%Y%m%d")
        end_dt = datetime.strptime(end, "%Y%m%d")
        chunks = []
        current = start_dt

        while current < end_dt:
            chunk_start = current.strftime("%Y%m%d")
            next_month = current.month + chunk_months
            year = current.year + (next_month - 1) // 12
            month = (next_month - 1) % 12 + 1
            chunk_end_dt = datetime(year, month, 1) - timedelta(days=1)
            chunk_end_dt = min(chunk_end_dt, end_dt)
            chunk_end = chunk_end_dt.strftime("%Y%m%d")
            chunks.append((chunk_start, chunk_end))
            current = chunk_end_dt + timedelta(days=1)

        return chunks

    def _read_locations_file(self, path: str) -> pd.DataFrame:
        """
        Read locations file and normalize columns.
        Required: loc, latitude, longitude, start_date, end_date
        """

        df = pd.read_csv(path)
        req = ["loc", "latitude", "longitude", "start_date", "end_date"]
        missing = [c for c in req if c not in df.columns]
        if missing:
            raise ValueError(f"Locations file missing columns: {missing}")

        # Parse dates like 9/24/25, 11/1/22, etc.
        sd = pd.to_datetime(df["start_date"], errors="coerce", infer_datetime_format=True)
        ed = pd.to_datetime(df["end_date"], errors="coerce", infer_datetime_format=True)

        if sd.isna().any() or ed.isna().any():
            bad = df[sd.isna() | ed.isna()][["loc", "start_date", "end_date"]].head(10)
            raise ValueError(f"Date parse failed for some rows. Examples:\n{bad}")

        out = df.copy()
        out["start_yyyymmdd"] = sd.dt.strftime("%Y%m%d")
        out["end_yyyymmdd"] = ed.dt.strftime("%Y%m%d")

        # Basic check
        sdt = pd.to_datetime(out["start_yyyymmdd"], format="%Y%m%d")
        edt = pd.to_datetime(out["end_yyyymmdd"], format="%Y%m%d")
        bad_range = out[sdt > edt]
        if not bad_range.empty:
            raise ValueError(f"Found start_date > end_date. Examples:\n{bad_range.head(10)}")

        return out



    def _query_single_task(self, args: Tuple[str, str, str]) -> pd.DataFrame:
        """
        Query weather data for a single task (used in parallel processing).

        Args:
            args: Tuple of (place_id, start_date, end_date)

        Returns:
            DataFrame with weather data
        """
        pid, start_date, end_date = args
        query = f"""
            SELECT place_id,
                lat AS centroid_lat,
                lon AS centroid_lon,
                date,
                temperature_c_2_m_above_gnd_max AS air_temp_max_c,
                temperature_c_2_m_above_gnd_min AS air_temp_min_c,
                temperature_c_2_m_above_gnd_avg AS air_temp_avg_c,
                temperature_c_0to10cm_below_gnd_avg AS soil_temp_c,
                precipitation_total_grid_mm_surface_sum AS precip_total_mm,
                relative_humidity_pct_2_m_above_gnd_avg AS humidity_mean_pct,
                soil_water_content_m3_per_m3_0to10cm_below_gnd_avg AS soil_moisture_mean_m3m3,
                daylight_duration_min_surface_sum AS sun_duration_min
            FROM spectrum_schema.mio171_weather_10x10_pivoted_archive
            WHERE place_id LIKE '{pid}'
                AND date >= '{start_date}'
                AND date <= '{end_date}'
            ORDER BY place_id, date
        """

        try:
            global conn
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return pd.DataFrame(result, columns=columns)
        except Exception as e:
            logger.error(f"Error querying task {args}: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error


if __name__ == "__main__":
    import os
    from pprint import pprint

    collector = WeatherDataCollector(DatabaseManager())
    print("Testing pull_single_location")


    # print("\nTesting _build_query:")
    # query = collector._build_query(
    #     latitude=40.7128,
    #     longitude=-74.0060,
    #     start_date="20240101",
    #     end_date="20240107",
    #     resolution="10by10",
    # )
    # print(query.strip())

    # result = collector.pull_single_location(
    #     location_id=1,
    #     latitude=40.7128,
    #     longitude=-74.0060,
    #     start_date="20250101",
    #     end_date="20250131",
    #     resolution="10by10"
    # )
    # print(f"number of rows: {len(result)}")

    # pprint(result.head().to_dict(orient="records"))

    locations_path = "./all_checks_weather_locations.csv"
    df = collector.pull_locations_parallel_full_rebuild_to_s3(
        locations_path=locations_path,
        resolution="10by10",
        chunk_months=2,
        processes=12,
        s3_bucket="sps-ds-bucket",
        s3_key="weeviltrak_data/historical_weather_data_10by10_updated_locations.h5",
    )
    print(f"rows: {len(df)}")
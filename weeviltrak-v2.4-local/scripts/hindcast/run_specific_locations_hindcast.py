"""
Run hindcast predictions for the specific locations defined in
``tests/predict_specific_locations.ipynb``.

The model is trained at runtime using data through 2024-12-31 inclusive, then
uploaded to S3 and used for the 2025 hindcast predictions.
"""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline.hindcast import HindcastPipeline
from app.pipeline.train_predict import WeevilTrakPipeline

CONFIG_PATH = "app/config/weeviltrak_v2.3.yml"
TRAIN_CUTOFF_DATE = "2024-12-31"
HINDCAST_START_DATE = "2025-03-01"
HINDCAST_END_DATE = "2025-12-31"
MODEL_PATH = "model_testing_2025/model_weeviltrak_lgbm_v2_3_trained_through_20241231.pkl"
OUTPUT_DIR = Path("outputs/hindcast/v2.3")

LOCATIONS = [
    {
        "name": "Landscape Management Research Center",
        "latitude": 40.80445835583023,
        "longitude": -77.86003957145041,
    },
    {
        "name": "2119 Farmington Road",
        "latitude": 40.309168907585125,
        "longitude": -75.65258900383164,
    },
    {
        "name": "Houston Oaks Golf Course",
        "latitude": 38.17201877100692,
        "longitude": -84.29430638975133,
    },
    {"name": "CDGA", "latitude": 41.6575, "longitude": -87.9550},
    {
        "name": "OSU Golf Club 3605 Tremont Rd",
        "latitude": 40.03205058474942,
        "longitude": -83.05298197316414,
    },
    {
        "name": "Heatherwoode Golf Course",
        "latitude": 39.53793661277499,
        "longitude": -84.23107635102484,
    },
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_dates = (
        pd.date_range(HINDCAST_START_DATE, HINDCAST_END_DATE, freq="D")
        .strftime("%Y-%m-%d")
        .tolist()
    )

    train_pipeline = WeevilTrakPipeline(CONFIG_PATH)
    train_pipeline.run_training(
        today=TRAIN_CUTOFF_DATE,
        include_today=True,
        model_path_override=MODEL_PATH,
    )

    pipeline = HindcastPipeline(CONFIG_PATH, output_dir=str(OUTPUT_DIR))
    out = pipeline.run_hindcast(
        target_dates=target_dates,
        locations=LOCATIONS,
        model_path_override=MODEL_PATH,
        save_local=True,
    )

    full_csv = OUTPUT_DIR / "predict_specific_locations_20250301_20251231_model20241231_full.csv"
    out_to_save = out.copy()
    out_to_save["prediction_date"] = pd.to_datetime(
        out_to_save["prediction_date"]
    ).dt.strftime("%Y-%m-%d")
    out_to_save["predicted_stage_date"] = pd.to_datetime(
        out_to_save["predicted_stage_date"]
    ).dt.strftime("%Y-%m-%d")
    out_to_save.to_csv(full_csv, index=False)

    latest = out.sort_values(["prediction_date", "input_name", "stage_id"]).copy()
    latest = latest[latest["prediction_date"] == latest["prediction_date"].max()].copy()
    latest["prediction_date"] = pd.to_datetime(latest["prediction_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    latest["predicted_stage_date"] = pd.to_datetime(
        latest["predicted_stage_date"]
    ).dt.strftime("%Y-%m-%d")
    latest_csv = OUTPUT_DIR / "predict_specific_locations_latest_2025-12-31_model20241231.csv"
    latest.to_csv(latest_csv, index=False)

    pivot = latest.pivot_table(
        index=[
            "input_name",
            "input_latitude",
            "input_longitude",
            "mapped_centroid_latitude",
            "mapped_centroid_longitude",
        ],
        columns="stage_id",
        values="predicted_stage_date",
        aggfunc="first",
    ).reset_index()
    pivot.columns = [
        f"Stage {int(c)}" if isinstance(c, (int, float)) else c
        for c in pivot.columns
    ]
    pivot_csv = OUTPUT_DIR / "predict_specific_locations_latest_2025-12-31_model20241231_pivot.csv"
    pivot.to_csv(pivot_csv, index=False)

    print("DONE")
    print(f"TRAIN_CUTOFF_DATE={TRAIN_CUTOFF_DATE}")
    print(f"MODEL_PATH={MODEL_PATH}")
    print(f"FULL_CSV={full_csv}")
    print(f"LATEST_CSV={latest_csv}")
    print(f"PIVOT_CSV={pivot_csv}")


if __name__ == "__main__":
    main()

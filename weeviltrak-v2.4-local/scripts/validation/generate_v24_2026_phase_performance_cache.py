"""Generate a reproducible current-v2.4 2026 Phase 1/Phase 2 trajectory cache.

This is an operational supplemental evaluation: the v2.4 artifacts are loaded
from S3 and are never trained on the 2026 Phase observations used for scoring.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.pipeline.backtesting import BacktestingFramework
from app.services.database_service import DatabaseManager


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "validation" / "v24_2026_phase_operational_trajectory.parquet"
OBSERVED_OUT = ROOT / "outputs" / "validation" / "v24_2026_phase_observed_events.parquet"
WEATHER = ROOT / "outputs" / "backtesting" / "v2.5" / "processed_weather_unique.parquet"


def load_current_observed_phases() -> pd.DataFrame:
    """Read the current Phase labels; do not rely on a stale local snapshot."""
    sql = """
        SELECT location_id, location_name, location_state, latitude, longitude,
               stage_name, stage_date
        FROM europe_dna.europe_dna_sps_weeviltrak
        WHERE EXTRACT(year FROM stage_date) = 2026
          AND stage_name IN ('Phase 1', 'Phase 2')
          AND TO_CHAR(stage_date, 'MM-DD') <> '12-31'
        ORDER BY location_id, stage_name, stage_date
    """
    with DatabaseManager() as database:
        observed = pd.read_sql(sql, database.get_connection())
    observed["stage_date"] = pd.to_datetime(observed["stage_date"], errors="coerce")
    observed["location_id"] = observed["location_id"].astype(str)
    # One verified date per location/phase is required for an error point.
    return observed.sort_values("stage_date").drop_duplicates(
        ["location_id", "stage_name"], keep="first"
    )


def main() -> None:
    observed = load_current_observed_phases()
    locations = sorted(observed["location_id"].astype(str).unique())
    if not locations:
        raise RuntimeError("No valid 2026 Phase 1/Phase 2 observations were found")
    print(
        "Current valid observed phase locations:",
        observed.groupby("stage_name")["location_id"].nunique().to_dict(),
    )

    weather = pd.read_parquet(WEATHER)
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
    weather = weather[
        weather["date"].dt.year.eq(2026)
        & weather["location_id"].astype(str).isin(locations)
    ].copy()
    if weather.empty:
        raise RuntimeError("No 2026 processed weather is available for phase locations")

    framework = BacktestingFramework(
        "app/config/weeviltrak_v2.4.yml", output_dir=str(ROOT / "outputs" / "validation" / "_v24_2026_phase"),
    )
    framework.model_manager.load_model()
    model = framework.model_manager.model
    prediction_dates = pd.date_range(weather["date"].min(), weather["date"].max(), freq="D")
    parts = []
    for date in prediction_dates:
        pred = framework.predict_with_model_for_date(
            model=model,
            test_date=date,
            locations_for_date=locations,
            model_train_cutoff_date="2026-01-01",
            processed_weather_unique=weather,
        )
        if pred is not None and not pred.empty:
            parts.append(pred[pred["stage_id"].isin([3, 4])].copy())
    if not parts:
        raise RuntimeError("No v2.4 Phase 1/Phase 2 trajectory rows were produced")

    trajectory = pd.concat(parts, ignore_index=True)
    trajectory = framework.aggregate_predictions_unique(trajectory)
    trajectory = framework._apply_optional_termination_to_predictions(trajectory)
    trajectory["target_name"] = trajectory["stage_id"].map({4: "Phase 1", 3: "Phase 2"})
    trajectory["evaluation_protocol"] = "2026_operational_artifact_supplemental"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    trajectory.to_parquet(OUT, index=False)
    observed.to_parquet(OBSERVED_OUT, index=False)
    print(f"Saved {len(trajectory):,} rows to {OUT}")
    print(f"Saved {len(observed):,} observed phase rows to {OBSERVED_OUT}")


if __name__ == "__main__":
    main()

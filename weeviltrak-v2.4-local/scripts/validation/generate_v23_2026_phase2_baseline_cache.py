"""Create a v2.3 first-zero baseline trajectory for 2026 Phase 2 comparison."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.pipeline.backtesting import BacktestingFramework


ROOT = Path(__file__).resolve().parents[2]
OBSERVED = ROOT / "outputs" / "validation" / "v24_2026_phase_observed_events.parquet"
WEATHER = ROOT / "outputs" / "backtesting" / "v2.5" / "processed_weather_unique.parquet"
OUT = ROOT / "outputs" / "validation" / "v23_2026_phase2_first_zero_trajectory.parquet"


def main() -> None:
    if not OBSERVED.exists():
        raise FileNotFoundError(f"Missing current Phase observations: {OBSERVED}")
    observed = pd.read_parquet(OBSERVED)
    locations = sorted(
        observed.loc[observed["stage_name"].eq("Phase 2"), "location_id"]
        .astype(str)
        .unique()
    )
    weather = pd.read_parquet(WEATHER)
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
    weather = weather[
        weather["date"].dt.year.eq(2026)
        & weather["location_id"].astype(str).isin(locations)
    ].copy()
    framework = BacktestingFramework(
        "app/config/weeviltrak_v2.3.yml", output_dir=str(ROOT / "outputs" / "validation" / "_v23_2026_phase2")
    )
    framework.model_manager.load_model()
    parts = []
    for date in pd.date_range(weather["date"].min(), weather["date"].max(), freq="D"):
        pred = framework.predict_with_model_for_date(
            model=framework.model_manager.model,
            test_date=date,
            locations_for_date=locations,
            model_train_cutoff_date="2026-01-01",
            processed_weather_unique=weather,
        )
        if pred is not None and not pred.empty:
            parts.append(pred[pred["stage_id"].eq(3)].copy())
    trajectory = framework.aggregate_predictions_unique(pd.concat(parts, ignore_index=True))
    trajectory["termination_reached"] = False
    trajectory["termination_decision_date"] = pd.NaT
    for _, group in trajectory.sort_values("prediction_date").groupby(["location_id", "stage_id"], sort=False):
        hit = group[pd.to_numeric(group["raw_signed_days"], errors="coerce").le(0)]
        if not hit.empty:
            decision = pd.to_datetime(hit.iloc[0]["prediction_date"])
            mask = trajectory.index.isin(group.index) & pd.to_datetime(trajectory["prediction_date"]).ge(decision)
            trajectory.loc[mask, "termination_reached"] = True
            trajectory.loc[mask, "termination_decision_date"] = decision
    OUT.parent.mkdir(parents=True, exist_ok=True)
    trajectory.to_parquet(OUT, index=False)
    print(f"Saved {len(trajectory):,} rows to {OUT}")


if __name__ == "__main__":
    main()

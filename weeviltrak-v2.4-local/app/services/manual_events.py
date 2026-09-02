"""Curated manual WeevilTrak event observations.

These rows come from mentor-provided slow/discrete observations documented in
``notebooks/experiments/manual_stage_data_validation_report.ipynb``.  They are
kept in code instead of Redshift so the production pipeline can opt into the
same records for training/validation without changing the source database.

Only clear, single-date Stage 1/2/3 observations are included.  Missing,
ambiguous, and interval observations remain evidence in the notebook, but are
not safe direct model labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


MANUAL_EVENT_SOURCE_FILE = "mentor_manual_notes_2025_2026"


@dataclass(frozen=True)
class ManualEventConfig:
    """Settings controlling curated manual event injection."""

    enabled: bool = False
    include_in_training: bool = True
    include_in_validation: bool = True

    @classmethod
    def from_config(cls, config: dict | None) -> "ManualEventConfig":
        raw = (config or {}).get("manual_events", {}) or {}
        return cls(
            enabled=bool(raw.get("enabled", False)),
            include_in_training=bool(raw.get("include_in_training", True)),
            include_in_validation=bool(raw.get("include_in_validation", True)),
        )


MANUAL_STAGE_EVENT_ROWS: tuple[dict, ...] = (
    {
        "region": "Chicago",
        "year": 2025,
        "abbrev": None,
        "location_name": "Glen Flora Country Club",
        "address": None,
        "latitude": 42.392092067115186,
        "longitude": -87.83332773512043,
        "stage_id": 1,
        "stage_name": "Stage 1",
        "stage_date": "2025-05-05",
        "evidence": "May 5th, 2025",
        "notes": "clear; Peak Activity of Overwintered Adults",
    },
    {
        "region": "Chicago",
        "year": 2025,
        "abbrev": None,
        "location_name": "Glen Flora Country Club",
        "address": None,
        "latitude": 42.392092067115186,
        "longitude": -87.83332773512043,
        "stage_id": 2,
        "stage_name": "Stage 2",
        "stage_date": "2025-05-18",
        "evidence": "May 18th, 2025",
        "notes": "clear; Peak Activity of Early Instar inside stem",
    },
    {
        "region": "Chicago",
        "year": 2025,
        "abbrev": None,
        "location_name": "Glen Flora Country Club",
        "address": None,
        "latitude": 42.392092067115186,
        "longitude": -87.83332773512043,
        "stage_id": 3,
        "stage_name": "Stage 3",
        "stage_date": "2025-06-01",
        "evidence": "June 1st, 2025",
        "notes": "clear; Peak Activity of large larvae outside stem",
    },
    {
        "region": "Kentucky",
        "year": 2026,
        "abbrev": "CT",
        "location_name": "Champion Trace",
        "address": "20 Ave of Champions, Nicholasville, KY 40356",
        "latitude": 37.973038,
        "longitude": -84.633743,
        "stage_id": 1,
        "stage_name": "Stage 1",
        "stage_date": "2026-04-02",
        "evidence": "CT - April 2nd",
        "notes": "clear; Champions Trace normalized to Champion Trace",
    },
    {
        "region": "Kentucky",
        "year": 2026,
        "abbrev": "HO",
        "location_name": "Houston Oaks",
        "address": "555 Houston Oaks Dr, Paris, KY 40361",
        "latitude": 38.171783,
        "longitude": -84.294525,
        "stage_id": 1,
        "stage_name": "Stage 1",
        "stage_date": "2026-04-02",
        "evidence": "HO - April 2nd",
        "notes": "clear",
    },
    {
        "region": "Kentucky",
        "year": 2026,
        "abbrev": "UL",
        "location_name": "University of Louisville Golf Club",
        "address": "401 Champions Way, Simpsonville, KY 40067",
        "latitude": 38.212247,
        "longitude": -85.359902,
        "stage_id": 1,
        "stage_name": "Stage 1",
        "stage_date": "2026-03-30",
        "evidence": "UL - March 30th",
        "notes": "clear",
    },
    {
        "region": "Kentucky",
        "year": 2026,
        "abbrev": "FCC",
        "location_name": "Frankfort Country Club",
        "address": "101 Duntreath St, Frankfort, KY 40601",
        "latitude": 38.204946,
        "longitude": -84.810389,
        "stage_id": 1,
        "stage_name": "Stage 1",
        "stage_date": "2026-04-13",
        "evidence": "FCC - April 13th",
        "notes": "clear",
    },
    {
        "region": "Kentucky",
        "year": 2026,
        "abbrev": "CT",
        "location_name": "Champion Trace",
        "address": "20 Ave of Champions, Nicholasville, KY 40356",
        "latitude": 37.973038,
        "longitude": -84.633743,
        "stage_id": 2,
        "stage_name": "Stage 2",
        "stage_date": "2026-04-23",
        "evidence": "CT - April 23rd",
        "notes": "clear; Peak Activity of Early Instar inside stem",
    },
    {
        "region": "Kentucky",
        "year": 2026,
        "abbrev": "UL",
        "location_name": "University of Louisville Golf Club",
        "address": "401 Champions Way, Simpsonville, KY 40067",
        "latitude": 38.212247,
        "longitude": -85.359902,
        "stage_id": 2,
        "stage_name": "Stage 2",
        "stage_date": "2026-04-27",
        "evidence": "UL - April 27th",
        "notes": "clear; Peak Activity of Early Instar inside stem",
    },
    {
        "region": "Kentucky",
        "year": 2026,
        "abbrev": "FCC",
        "location_name": "Frankfort Country Club",
        "address": "101 Duntreath St, Frankfort, KY 40601",
        "latitude": 38.204946,
        "longitude": -84.810389,
        "stage_id": 2,
        "stage_name": "Stage 2",
        "stage_date": "2026-05-11",
        "evidence": "FCC - smaller peak on April 20th, biggest peak on May 11th",
        "notes": "clear main peak; smaller peak 2026-04-20 kept in notes",
    },
    {
        "region": "Kentucky",
        "year": 2026,
        "abbrev": "FCC",
        "location_name": "Frankfort Country Club",
        "address": "101 Duntreath St, Frankfort, KY 40601",
        "latitude": 38.204946,
        "longitude": -84.810389,
        "stage_id": 3,
        "stage_name": "Stage 3",
        "stage_date": "2026-06-08",
        "evidence": "FCC - June 8th; moderate activity before then",
        "notes": "clear main peak, moderate activity before",
    },
)


def _slug(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def load_manual_stage_events() -> pd.DataFrame:
    """Return curated manual Stage 1/2/3 rows aligned to Redshift-style fields."""

    df = pd.DataFrame(MANUAL_STAGE_EVENT_ROWS).copy()
    if df.empty:
        return df

    df["source_file"] = MANUAL_EVENT_SOURCE_FILE
    df["stage_date"] = pd.to_datetime(df["stage_date"], errors="raise")
    df["year"] = df["stage_date"].dt.year.astype("int64")
    df["day_of_year"] = df["stage_date"].dt.dayofyear
    df["location_id"] = (
        "manual_"
        + df["region"].map(_slug)
        + "_"
        + df["year"].astype(str)
        + "_"
        + df["location_name"].map(_slug)
    )
    df["manual_event_id"] = (
        df["location_id"] + "_stage_" + df["stage_id"].astype(str)
    )
    df["data_source"] = "manual_curated"
    df["is_manual_event"] = True
    return df


def append_manual_stage_events(
    weevil_data: pd.DataFrame,
    *,
    config: ManualEventConfig | None = None,
) -> pd.DataFrame:
    """Append curated manual rows to processed Redshift rows when enabled."""

    settings = config or ManualEventConfig(enabled=True)
    if not settings.enabled or not settings.include_in_training:
        return weevil_data

    manual = load_manual_stage_events()
    if manual.empty:
        return weevil_data

    base = weevil_data.copy()
    base["is_manual_event"] = base.get("is_manual_event", False)
    if "data_source" not in base.columns:
        base["data_source"] = "redshift"

    cols = list(dict.fromkeys(base.columns.tolist() + manual.columns.tolist()))
    out = pd.concat(
        [base.reindex(columns=cols), manual.reindex(columns=cols)],
        ignore_index=True,
        sort=False,
    )
    return out

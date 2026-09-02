"""Canonical WeevilTrak event mapping utilities.

This module isolates business event semantics from the raw Redshift
``stage_id`` / ``stage_name`` fields.  The source system reused numeric
``stage_id`` values when 2026 data moved from Stage labels to Phase labels, so
the application must not treat raw IDs as stable model targets.

The production Stage/Phase model predicts four stable business events:

* Stage 1
* Stage 2
* Stage 3 / Phase 2
* Phase 1, represented by a derived midpoint(Stage 1, Stage 2) training label

Observed Phase 1 is deliberately not used to invent missing Stage 1/2 labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


MODEL_STAGE_IDS = (1, 4, 2, 3)

STAGE_OUTPUT_NAMES = {
    1: "Stage 1",
    2: "Stage 2",
    3: "Stage 3 / Phase 2",
    4: "Phase 1",
}

BUSINESS_OUTPUT_ORDER = ("stage_1", "phase_1", "stage_2", "stage_3_phase_2")

BUSINESS_OUTPUT_NAMES = {
    "stage_1": "Stage 1",
    "phase_1": "Phase 1",
    "stage_2": "Stage 2",
    "stage_3_phase_2": "Stage 3 / Phase 2",
}

BUSINESS_OUTPUT_STAGE_IDS = {
    "stage_1": 1,
    # Synthetic canonical ID.  This is intentionally not Redshift's raw
    # ``stage_id`` because in 2026 raw stage_id=2 means Phase 1, while in older
    # data raw stage_id=2 meant Stage 1.
    "phase_1": 4,
    "stage_2": 2,
    "stage_3_phase_2": 3,
}


@dataclass(frozen=True)
class CanonicalEventConfig:
    """Settings controlling Stage/Phase canonicalization."""

    enabled: bool = False
    include_phase2_as_stage3: bool = True
    include_observed_phase1: bool = False
    exclude_placeholder_dates: bool = True
    placeholder_month_day: str = "12-31"
    include_derived_phase1: bool = True

    @classmethod
    def from_config(cls, config: dict | None) -> "CanonicalEventConfig":
        raw = (config or {}).get("canonical_events", {}) or {}
        return cls(
            enabled=bool(raw.get("enabled", False)),
            include_phase2_as_stage3=bool(raw.get("include_phase2_as_stage3", True)),
            include_observed_phase1=bool(raw.get("include_observed_phase1", False)),
            exclude_placeholder_dates=bool(raw.get("exclude_placeholder_dates", True)),
            placeholder_month_day=str(raw.get("placeholder_month_day", "12-31")),
            include_derived_phase1=bool(raw.get("include_derived_phase1", True)),
        )


def canonical_events_enabled(config: dict | None) -> bool:
    return CanonicalEventConfig.from_config(config).enabled


def _normalize_name(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _is_placeholder_date(series: pd.Series, month_day: str = "12-31") -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce")
    return dates.dt.strftime("%m-%d").eq(month_day)


def build_canonical_training_events(
    raw_events: pd.DataFrame,
    *,
    config: CanonicalEventConfig | None = None,
) -> pd.DataFrame:
    """Map raw Redshift events into stable model training events.

    Parameters
    ----------
    raw_events:
        Raw rows from ``europe_dna.europe_dna_sps_weeviltrak``.
    config:
        Canonical event settings.  Defaults to enabled canonical behavior.

    Returns
    -------
    pandas.DataFrame
        Rows with ``stage_id`` rewritten as the stable model event ID:
        1 = Stage 1, 2 = Stage 2, 3 = Stage 3 / Phase 2.
    """

    settings = config or CanonicalEventConfig(enabled=True)
    if raw_events is None or raw_events.empty:
        return pd.DataFrame(
            columns=list(raw_events.columns if raw_events is not None else [])
        )

    df = raw_events.copy()
    df["stage_date"] = pd.to_datetime(df["stage_date"], errors="coerce")
    df["stage_name_raw"] = df["stage_name"]
    df["raw_stage_id"] = df.get("stage_id")
    normalized = df["stage_name"].map(_normalize_name)

    canonical_stage_id = pd.Series(np.nan, index=df.index, dtype="float")
    label_source = pd.Series(pd.NA, index=df.index, dtype="object")

    canonical_stage_id.loc[normalized.eq("stage 1")] = 1
    label_source.loc[normalized.eq("stage 1")] = "observed_stage1"

    canonical_stage_id.loc[normalized.eq("stage 2")] = 2
    label_source.loc[normalized.eq("stage 2")] = "observed_stage2"

    canonical_stage_id.loc[normalized.eq("stage 3")] = 3
    label_source.loc[normalized.eq("stage 3")] = "observed_stage3"

    if settings.include_phase2_as_stage3:
        phase2_mask = normalized.eq("phase 2")
        canonical_stage_id.loc[phase2_mask] = 3
        label_source.loc[phase2_mask] = "observed_phase2_as_stage3"

    # A release may opt into using quality-controlled observed Phase 1 dates
    # as direct canonical Phase 1 labels.  This never fabricates Stage 1 or
    # Stage 2 labels; it only contributes to the stable Phase 1 target (ID 4).
    if settings.include_observed_phase1:
        phase1_mask = normalized.eq("phase 1")
        canonical_stage_id.loc[phase1_mask] = 4
        label_source.loc[phase1_mask] = "observed_phase1"

    df["stage_id"] = canonical_stage_id
    df["canonical_event_id"] = canonical_stage_id
    df["canonical_event_name"] = df["stage_id"].map(STAGE_OUTPUT_NAMES)
    df["label_source"] = label_source

    df = df[df["stage_id"].notna()].copy()
    if settings.exclude_placeholder_dates:
        df = df[
            ~_is_placeholder_date(df["stage_date"], settings.placeholder_month_day)
        ].copy()

    df["stage_id"] = df["stage_id"].astype(int)
    df["canonical_event_id"] = df["canonical_event_id"].astype(int)
    df["stage_name"] = df["canonical_event_name"]
    return df


def append_phase1_midpoint_training_events(
    events: pd.DataFrame,
    *,
    config: CanonicalEventConfig | None = None,
) -> pd.DataFrame:
    """Append direct Phase 1 labels derived from paired Stage 1/Stage 2 events.

    A pseudo label is created only for one clear Stage 1 and one clear Stage 2
    event at the same location in the same calendar year.  The midpoint is kept
    at timestamp precision so a half-day midpoint is not silently rounded.
    """
    settings = config or CanonicalEventConfig(enabled=True)
    if not settings.include_derived_phase1 or events is None or events.empty:
        return events

    base = events.copy()
    base["stage_date"] = pd.to_datetime(base["stage_date"], errors="coerce")
    if "year" not in base.columns:
        base["year"] = base["stage_date"].dt.year
    base["location_id"] = base["location_id"].astype(str)
    direct = base[base["stage_id"].isin([1, 2]) & base["stage_date"].notna()].copy()
    if direct.empty:
        return base

    group_cols = ["location_id", "year"]
    pairs = (
        direct.sort_values("stage_date")
        .drop_duplicates(group_cols + ["stage_id"], keep="first")
        .pivot(index=group_cols, columns="stage_id", values="stage_date")
        .reset_index()
    )
    if 1 not in pairs.columns or 2 not in pairs.columns:
        return base
    pairs = pairs.dropna(subset=[1, 2]).copy()
    if pairs.empty:
        return base

    lookup = (
        direct.sort_values("stage_date")
        .drop_duplicates(group_cols, keep="first")
        .set_index(group_cols)
    )
    rows = []
    for pair in pairs.itertuples(index=False):
        location_id, year, stage1_date, stage2_date = pair
        source = lookup.loc[(str(location_id), year)].copy()
        # location_id/year are index levels in ``lookup``; restore them as
        # ordinary columns before constructing the derived event row.
        source["location_id"] = str(location_id)
        source["year"] = int(year)
        source["stage_date"] = (
            pd.Timestamp(stage1_date)
            + (pd.Timestamp(stage2_date) - pd.Timestamp(stage1_date)) / 2
        )
        source["stage_id"] = 4
        source["canonical_event_id"] = 4
        source["stage_name"] = "Phase 1"
        source["canonical_event_name"] = "Phase 1"
        source["stage_name_raw"] = "Derived Phase 1 midpoint"
        source["label_source"] = "derived_phase1_midpoint_stage1_stage2"
        source["is_derived_phase1"] = True
        source["phase1_stage1_date"] = pd.Timestamp(stage1_date)
        source["phase1_stage2_date"] = pd.Timestamp(stage2_date)
        source["day_of_year"] = source["stage_date"].dayofyear
        rows.append(source.to_dict())
    phase1 = pd.DataFrame.from_records(rows)
    if "is_derived_phase1" not in base.columns:
        base["is_derived_phase1"] = False
    return pd.concat([base, phase1], ignore_index=True, sort=False)


def add_stage_output_names(predictions: pd.DataFrame) -> pd.DataFrame:
    """Attach stable business labels to direct model Stage 1/2/3 rows."""

    if predictions is None or predictions.empty:
        return predictions
    out = predictions.copy()
    numeric_stage_id = pd.to_numeric(out["stage_id"], errors="coerce")
    out["model_stage_id"] = numeric_stage_id.astype("Int64")
    out["output_event_id"] = numeric_stage_id.map(
        {1: "stage_1", 4: "phase_1", 2: "stage_2", 3: "stage_3_phase_2"}
    )
    out["output_event_name"] = out["output_event_id"].map(BUSINESS_OUTPUT_NAMES)
    out["output_event_order"] = out["output_event_id"].map(
        {event_id: idx + 1 for idx, event_id in enumerate(BUSINESS_OUTPUT_ORDER)}
    )
    return out


def derive_phase1_from_stage_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Append Phase 1 rows derived from Stage 1/2 predicted dates.

    Phase 1 is defined as midpoint(Stage 1, Stage 2).  The function keeps the
    original Stage 1/2/3 rows and appends one derived Phase 1 row per
    prediction-date/location when both Stage 1 and Stage 2 dates exist.
    """

    if predictions is None or predictions.empty:
        return predictions

    base = add_stage_output_names(predictions)
    # Current v2.4 predicts Phase 1 directly.  Keep this helper as a backwards
    # compatible fallback for older three-target prediction frames.
    if pd.to_numeric(base["stage_id"], errors="coerce").eq(4).any():
        base["is_derived_event"] = False
        base["derivation_rule"] = pd.NA
        return base
    df = base.copy()
    df["predicted_stage_date"] = pd.to_datetime(
        df["predicted_stage_date"], errors="coerce"
    )
    df["prediction_date"] = pd.to_datetime(df["prediction_date"], errors="coerce")

    group_cols = [
        col
        for col in [
            "prediction_date",
            "location_id",
            "place_id",
            "latitude",
            "longitude",
        ]
        if col in df.columns
    ]
    if "prediction_date" not in group_cols:
        raise ValueError("prediction_date is required to derive Phase 1")

    stage_dates = (
        df[df["stage_id"].isin([1, 2])]
        .pivot_table(
            index=group_cols,
            columns="stage_id",
            values="predicted_stage_date",
            aggfunc="first",
        )
        .reset_index()
    )
    if 1 not in stage_dates.columns or 2 not in stage_dates.columns:
        return base

    phase = stage_dates.dropna(subset=[1, 2]).copy()
    if phase.empty:
        return base

    midpoint = (
        pd.to_datetime(phase[1])
        + (pd.to_datetime(phase[2]) - pd.to_datetime(phase[1])) / 2
    )
    phase_rows = phase[group_cols].copy()
    phase_rows["stage_id"] = BUSINESS_OUTPUT_STAGE_IDS["phase_1"]
    phase_rows["model_stage_id"] = pd.NA
    phase_rows["output_event_id"] = "phase_1"
    phase_rows["output_event_name"] = BUSINESS_OUTPUT_NAMES["phase_1"]
    phase_rows["output_event_order"] = 2
    phase_rows["prediction_date"] = pd.to_datetime(phase_rows["prediction_date"])
    phase_rows["predicted_stage_date"] = midpoint.dt.normalize()
    phase_rows["predicted_days"] = (
        phase_rows["predicted_stage_date"] - phase_rows["prediction_date"]
    ).dt.days.astype(float)
    phase_rows["raw_signed_days"] = pd.NA
    phase_rows["is_derived_event"] = True
    phase_rows["derivation_rule"] = "midpoint(predicted Stage 1, predicted Stage 2)"

    base["is_derived_event"] = False
    base["derivation_rule"] = pd.NA
    base["prediction_date"] = pd.to_datetime(base["prediction_date"], errors="coerce")
    base["predicted_stage_date"] = pd.to_datetime(
        base["predicted_stage_date"], errors="coerce"
    )
    combined = pd.concat([base, phase_rows], ignore_index=True, sort=False)
    combined["output_event_order"] = pd.to_numeric(
        combined["output_event_order"], errors="coerce"
    )
    combined = combined.sort_values(
        [
            col
            for col in [
                "prediction_date",
                "location_id",
                "place_id",
                "output_event_order",
            ]
            if col in combined.columns
        ]
    ).reset_index(drop=True)
    combined["predicted_stage_date"] = pd.to_datetime(
        combined["predicted_stage_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    combined["prediction_date"] = pd.to_datetime(
        combined["prediction_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    return combined


def model_stage_ids_from_config(config: dict | None) -> Iterable[int]:
    """Return direct model stage IDs for prediction expansion."""

    if not canonical_events_enabled(config):
        return MODEL_STAGE_IDS
    raw = ((config or {}).get("canonical_events", {}) or {}).get("model_stage_ids")
    if raw:
        return tuple(int(value) for value in raw)
    return MODEL_STAGE_IDS

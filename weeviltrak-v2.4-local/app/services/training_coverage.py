"""Recent-label coverage checks for safe rolling-window retraining."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RecentLabelCoveragePolicy:
    """Minimum direct-label coverage required before replacing a model artifact."""

    enabled: bool = False
    min_events_per_stage: int = 20
    min_distinct_years_per_stage: int = 2
    on_insufficient: str = "block_retraining"

    @classmethod
    def from_config(cls, config: dict | None) -> "RecentLabelCoveragePolicy":
        raw = ((config or {}).get("recent_label_coverage", {}) or {})
        policy = cls(
            enabled=bool(raw.get("enabled", False)),
            min_events_per_stage=int(raw.get("min_events_per_stage", 20)),
            min_distinct_years_per_stage=int(
                raw.get("min_distinct_years_per_stage", 2)
            ),
            on_insufficient=str(raw.get("on_insufficient", "block_retraining")),
        )
        if policy.min_events_per_stage <= 0 or policy.min_distinct_years_per_stage <= 0:
            raise ValueError("recent_label_coverage minimums must be positive")
        if policy.on_insufficient != "block_retraining":
            raise ValueError(
                "recent_label_coverage.on_insufficient must be 'block_retraining'"
            )
        return policy


def summarize_recent_label_coverage(
    events: pd.DataFrame,
    *,
    cutoff_date: str,
    window_years: int,
    stage_ids: tuple[int, ...],
) -> pd.DataFrame:
    """Summarize direct event labels inside a rolling training window."""
    cutoff = pd.Timestamp(cutoff_date).normalize()
    start = cutoff - pd.DateOffset(years=window_years)
    frame = events.copy()
    frame["stage_date"] = pd.to_datetime(frame["stage_date"], errors="coerce")
    frame["stage_id"] = pd.to_numeric(frame["stage_id"], errors="coerce")
    frame = frame[
        frame["stage_date"].ge(start)
        & frame["stage_date"].lt(cutoff)
        & frame["stage_id"].isin(stage_ids)
    ].copy()
    frame["event_year"] = frame["stage_date"].dt.year

    summary = (
        frame.groupby("stage_id", as_index=False)
        .agg(
            n_events=("stage_date", "size"),
            n_distinct_years=("event_year", "nunique"),
            n_distinct_locations=("location_id", "nunique"),
        )
        .set_index("stage_id")
    )
    summary = summary.reindex(stage_ids, fill_value=0).rename_axis("stage_id")
    summary = summary.reset_index()
    summary["window_start"] = start.date().isoformat()
    summary["cutoff_date"] = cutoff.date().isoformat()
    return summary


def require_recent_label_coverage(
    events: pd.DataFrame,
    *,
    cutoff_date: str,
    window_years: int | None,
    stage_ids: tuple[int, ...],
    policy: RecentLabelCoveragePolicy,
) -> pd.DataFrame:
    """Raise before model saving when the configured recent window is sparse."""
    if not policy.enabled or window_years is None:
        return pd.DataFrame()

    summary = summarize_recent_label_coverage(
        events,
        cutoff_date=cutoff_date,
        window_years=window_years,
        stage_ids=stage_ids,
    )
    insufficient = summary[
        (summary["n_events"] < policy.min_events_per_stage)
        | (summary["n_distinct_years"] < policy.min_distinct_years_per_stage)
    ]
    if not insufficient.empty:
        raise RuntimeError(
            "Recent label coverage is insufficient for safe retraining. "
            "No model artifact was replaced; retain the last approved artifact and "
            "run prediction-only until new direct Stage labels are available. "
            f"Coverage: {summary.to_dict(orient='records')}"
        )
    return summary

"""Small local inference interface for a delivered WeevilTrak v2.4 bundle.

The helper deliberately loads local pickle files only. It does not require S3,
Redshift, or training data at prediction time.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from app.pipeline.termination import apply_learned_termination_rule
from app.services.canonical_events import add_stage_output_names


@dataclass
class LocalWeevilTrakRelease:
    """A validated local Base + termination model release."""

    config: dict
    manifest: dict
    base_model: object
    termination_model: object

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "LocalWeevilTrakRelease":
        artifact_dir = Path(artifact_dir)
        required = {
            "config": artifact_dir / "config.yml",
            "manifest": artifact_dir / "manifest.json",
            "base model": artifact_dir / "base_model.pkl",
            "termination model": artifact_dir / "termination_model.pkl",
        }
        missing = [name for name, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Incomplete release bundle at {artifact_dir}: missing {missing}"
            )
        with required["config"].open() as handle:
            config = yaml.safe_load(handle)
        manifest = json.loads(required["manifest"].read_text())
        with required["base model"].open("rb") as handle:
            base_model = pickle.load(handle)
        with required["termination model"].open("rb") as handle:
            termination_model = pickle.load(handle)
        release = cls(config, manifest, base_model, termination_model)
        release._validate_artifacts()
        return release

    @property
    def features(self) -> list[str]:
        return list(self.config["features"])

    @property
    def stage_ids(self) -> tuple[int, ...]:
        return tuple(self.config["canonical_events"]["model_stage_ids"])

    def _validate_artifacts(self) -> None:
        expected = self.features
        actual = list(getattr(self.base_model, "feature_name_", []))
        if actual and actual != expected:
            raise ValueError(
                "Base model feature schema differs from config.yml: "
                f"expected={expected}, actual={actual}"
            )
        if not hasattr(self.base_model, "predict"):
            raise TypeError("base_model.pkl does not provide predict()")
        if not hasattr(self.termination_model, "predict"):
            raise TypeError("termination_model.pkl does not provide predict()")

    def predict_trajectory(self, daily_features: pd.DataFrame) -> pd.DataFrame:
        """Predict a full visible daily trajectory using local artifacts.

        ``daily_features`` must contain the configured feature columns plus
        ``prediction_date``, ``latitude``, and ``longitude``. The optional
        location/place identifiers are preserved in output.
        """
        frame = daily_features.copy()
        required = set(self.features) | {"prediction_date", "latitude", "longitude"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")
        frame["prediction_date"] = pd.to_datetime(frame["prediction_date"], errors="coerce")
        if frame["prediction_date"].isna().any():
            raise ValueError("prediction_date must contain valid ISO dates")
        frame["stage_id"] = pd.to_numeric(frame["stage_id"], errors="coerce")
        if frame["stage_id"].isna().any():
            raise ValueError("stage_id must be numeric")
        frame["stage_id"] = frame["stage_id"].astype(int)
        unsupported = sorted(set(frame["stage_id"]) - set(self.stage_ids))
        if unsupported:
            raise ValueError(
                f"Unsupported stage_id values {unsupported}; expected {self.stage_ids}"
            )
        for feature in self.features:
            frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
        if frame[self.features].isna().any().any():
            missing_features = frame[self.features].columns[frame[self.features].isna().any()].tolist()
            raise ValueError(f"Required model features contain missing/non-numeric values: {missing_features}")

        sort_cols = ["prediction_date"]
        if "location_id" in frame.columns:
            sort_cols.append("location_id")
        sort_cols.append("stage_id")
        frame = frame.sort_values(sort_cols).reset_index(drop=True)
        raw_signed_days = np.asarray(self.base_model.predict(frame[self.features]), dtype=float)
        out_columns = ["stage_id", "latitude", "longitude"]
        out_columns.extend(col for col in ("place_id", "location_id") if col in frame.columns)
        output = frame[out_columns].copy()
        output["prediction_date"] = frame["prediction_date"].to_numpy()
        output["raw_signed_days"] = raw_signed_days
        output["predicted_days"] = np.clip(raw_signed_days, a_min=0.0, a_max=None)
        output["predicted_stage_date"] = (
            output["prediction_date"] + pd.to_timedelta(output["predicted_days"], unit="D")
        ).dt.strftime("%Y-%m-%d")
        output["trajectory_year"] = output["prediction_date"].dt.year

        policy = self.config.get("termination_model", {})
        final = apply_learned_termination_rule(
            predictions=output,
            model=self.termination_model,
            threshold=float(policy.get("threshold", 4.5)),
            overwrite_business_prediction=bool(policy.get("overwrite_business_prediction", True)),
            min_history_days=int(policy.get("min_history_days", 7)),
        )
        return add_stage_output_names(final)

    def predict_today(self, daily_features: pd.DataFrame) -> pd.DataFrame:
        """Return the final available run date from a visible seasonal input."""
        trajectory = self.predict_trajectory(daily_features)
        latest = pd.to_datetime(trajectory["prediction_date"]).max()
        return trajectory.loc[
            pd.to_datetime(trajectory["prediction_date"]).eq(latest)
        ].sort_values("output_event_order").reset_index(drop=True)

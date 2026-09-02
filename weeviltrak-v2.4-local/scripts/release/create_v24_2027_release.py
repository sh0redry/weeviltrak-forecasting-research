"""Build and publish an immutable v2.4 production release for the 2027 season.

The release has two serialized estimators: the base LightGBM model and the
learned Ridge termination model.  A resolved YAML configuration and JSON
manifest are published beside them.  Daily 2027 jobs use the resolved YAML in
prediction-only mode; they never retrain or mutate these artifacts.

Example:
    poetry run python scripts/release/create_v24_2027_release.py \
        --cutoff-date 2026-08-03 \
        --release-id v2.4-2027-20260803 \
        --train-and-publish
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from app.pipeline.train_predict import WeevilTrakPipeline
from app.services.canonical_events import model_stage_ids_from_config
from app.services.training_coverage import (
    RecentLabelCoveragePolicy,
    require_recent_label_coverage,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = ROOT / "app/config/weeviltrak_v2.4_2027_release.yml"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/model_releases"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _assert_absent(s3, keys: list[str]) -> None:
    existing = [key for key in keys if s3.file_exists(key)]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite an existing model release. Existing keys: "
            + ", ".join(existing)
        )


def _validate_loaded_artifacts(pipeline: WeevilTrakPipeline, config: dict) -> dict:
    """Load the serialized models and verify the base feature contract."""
    base = pipeline.s3_manager.read_file(config["model_path"])
    termination = pipeline.s3_manager.read_file(config["termination_model"]["model_path"])
    if base is None or termination is None:
        raise RuntimeError("Published model artifact could not be loaded")

    actual_features = list(getattr(base, "feature_name_", []))
    expected_features = list(config["features"])
    if actual_features and actual_features != expected_features:
        raise RuntimeError(
            "Base model feature schema does not match the frozen release config: "
            f"expected={expected_features}, actual={actual_features}"
        )
    if not hasattr(termination, "predict"):
        raise RuntimeError("Termination artifact is not a fitted prediction model")
    return {
        "base_model_class": type(base).__name__,
        "termination_model_class": type(termination).__name__,
        "base_model_feature_names": actual_features or expected_features,
    }


def _build_release_config(template: dict, *, release_prefix: str, output_path: str) -> dict:
    config = json.loads(json.dumps(template))
    config["model_path"] = f"{release_prefix}/base_model.pkl"
    config["hindcast_model_path"] = config["model_path"]
    config["termination_model"]["model_path"] = f"{release_prefix}/termination_model.pkl"
    config["output_path"] = output_path
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a v2.4 2027 model-release bundle")
    parser.add_argument("--cutoff-date", required=True, help="Inclusive training-data cutoff, YYYY-MM-DD")
    parser.add_argument("--release-id", required=True, help="Immutable release identifier, e.g. v2.4-2027-20260803")
    parser.add_argument("--template-config", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--train-and-publish",
        action="store_true",
        help="Train artifacts in a staging prefix, validate them, then publish the release prefix",
    )
    args = parser.parse_args()

    cutoff = pd.Timestamp(args.cutoff_date).normalize()
    if cutoff > pd.Timestamp.today().normalize():
        raise ValueError("cutoff-date cannot be in the future")
    if not args.template_config.is_file():
        raise FileNotFoundError(args.template_config)

    template = yaml.safe_load(args.template_config.read_text(encoding="utf-8"))
    canonical = template.get("canonical_events", {})
    if not canonical.get("include_observed_phase1", False):
        raise ValueError("2027 release template must enable observed Phase 1 labels")
    if int(template.get("training_window_years", 0)) != 3:
        raise ValueError("2027 release must use the agreed three-year training window")

    release_dir = args.output_root / args.release_id
    release_dir.mkdir(parents=True, exist_ok=True)
    s3_root = f"weeviltrak_data/model-releases/{args.release_id}"
    staging_prefix = f"{s3_root}/staging"
    final_prefix = f"{s3_root}/release"
    output_path = f"weeviltrak_data/api-testing/{args.release_id}"

    staging_config = _build_release_config(template, release_prefix=staging_prefix, output_path=output_path)
    final_config = _build_release_config(template, release_prefix=final_prefix, output_path=output_path)
    staging_config_path = release_dir / "staging_config.yml"
    final_config_path = release_dir / "config.yml"
    _write_yaml(staging_config_path, staging_config)
    _write_yaml(final_config_path, final_config)

    pipeline = WeevilTrakPipeline(config_path=str(staging_config_path))
    final_keys = [
        final_config["model_path"],
        final_config["termination_model"]["model_path"],
        f"{final_prefix}/config.yml",
        f"{final_prefix}/manifest.json",
    ]
    _assert_absent(pipeline.s3_manager, final_keys)

    events = pipeline.data_service.pull_weevil_data(today=cutoff.strftime("%Y-%m-%d"))
    stage_ids = tuple(model_stage_ids_from_config(staging_config))
    coverage = require_recent_label_coverage(
        events,
        cutoff_date=cutoff.strftime("%Y-%m-%d"),
        window_years=3,
        stage_ids=stage_ids,
        policy=RecentLabelCoveragePolicy.from_config(staging_config),
    )
    coverage.to_csv(release_dir / "label_coverage.csv", index=False)
    if "label_source" not in events.columns:
        events["label_source"] = "unknown"
    label_counts = (
        events.assign(label_source=events["label_source"].fillna("unknown"))
        .groupby(["stage_id", "label_source"], dropna=False)
        .size()
        .reset_index(name="n_events")
        .to_dict(orient="records")
    )

    if not args.train_and_publish:
        print("Preflight passed. Re-run with --train-and-publish to create artifacts.")
        print(f"Local release directory: {release_dir}")
        return

    staging_keys = [
        staging_config["model_path"],
        staging_config["termination_model"]["model_path"],
    ]
    _assert_absent(pipeline.s3_manager, staging_keys)
    pipeline.run_training(today=cutoff.strftime("%Y-%m-%d"), window_years=3)
    staging_validation = _validate_loaded_artifacts(pipeline, staging_config)

    # Copy validated artifacts to the immutable release prefix; S3-side copy
    # avoids changing pickle bytes after validation.
    for source_key, destination_key in zip(staging_keys, final_keys[:2]):
        pipeline.s3_manager.client.copy_object(
            Bucket=pipeline.s3_manager.bucket_name,
            Key=destination_key,
            CopySource={"Bucket": pipeline.s3_manager.bucket_name, "Key": source_key},
        )

    final_validation = _validate_loaded_artifacts(
        WeevilTrakPipeline(config_path=str(final_config_path)), final_config
    )
    manifest = {
        "release_id": args.release_id,
        "status": "published",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_cutoff_date": cutoff.date().isoformat(),
        "training_window_years": 3,
        "canonical_output_targets": ["Stage 1", "Phase 1", "Stage 2", "Stage 3 / Phase 2"],
        "phase1_label_policy": {
            "derived_midpoint_stage1_stage2": True,
            "observed_phase1_enabled": True,
            "observed_phase1_never_backfilled_to_stage1_or_stage2": True,
        },
        "phase2_label_policy": "Observed Phase 2 maps to Stage 3 / Phase 2",
        "manual_events_enabled": bool(template.get("manual_events", {}).get("enabled")),
        "features": final_config["features"],
        "label_coverage": coverage.to_dict(orient="records"),
        "label_counts_by_source": label_counts,
        "artifacts": {
            "base_model": final_config["model_path"],
            "termination_model": final_config["termination_model"]["model_path"],
            "config": f"{final_prefix}/config.yml",
        },
        "validation": {"staging": staging_validation, "final": final_validation},
        "template_config_sha256": _sha256_file(args.template_config),
        "resolved_config_sha256": _sha256_file(final_config_path),
        "daily_command": (
            "poetry run python -m app.pipeline.predict_only "
            f"--config-path {final_config_path} --test-date 2027-03-01"
        ),
    }
    manifest_path = release_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    pipeline.s3_manager.upload_file(final_config_path.read_text(encoding="utf-8"), f"{final_prefix}/config.yml")
    pipeline.s3_manager.upload_file(manifest_path.read_text(encoding="utf-8"), f"{final_prefix}/manifest.json")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()

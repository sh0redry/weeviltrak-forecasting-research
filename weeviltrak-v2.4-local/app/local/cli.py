"""Command-line interface for the offline WeevilTrak v2.4 workflow."""

from __future__ import annotations

import argparse
import json

from app.local.pipeline import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_CONFIG,
    DEFAULT_EVENTS,
    DEFAULT_FEATURES,
    add_configured_manual_events,
    generate_mock_engineered_features,
    load_events,
    predict_local,
    train_local,
    walk_forward_local,
)
from app.config.weevilltrak_config import WeevillTrakConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WeevilTrak v2.4 local workflow")
    commands = parser.add_subparsers(dest="command", required=True)

    mock = commands.add_parser("prepare-mock", help="Generate synthetic smoke-test features")
    mock.add_argument("--events", default=str(DEFAULT_EVENTS))
    mock.add_argument("--output", default=str(DEFAULT_FEATURES))

    train = commands.add_parser("train", help="Train local Base and termination artifacts")
    train.add_argument("--config", default=str(DEFAULT_CONFIG))
    train.add_argument("--events", default=str(DEFAULT_EVENTS))
    train.add_argument("--features", default=str(DEFAULT_FEATURES))
    train.add_argument("--artifacts", default=str(DEFAULT_ARTIFACT_DIR))

    predict = commands.add_parser("predict", help="Generate predictions for one date")
    predict.add_argument("--date", required=True)
    predict.add_argument("--events", default=str(DEFAULT_EVENTS))
    predict.add_argument("--features", default=str(DEFAULT_FEATURES))
    predict.add_argument("--artifacts", default=str(DEFAULT_ARTIFACT_DIR))
    predict.add_argument("--output", default="outputs/local/predictions.csv")

    validate = commands.add_parser("walk-forward", help="Run chronological validation")
    validate.add_argument("--config", default=str(DEFAULT_CONFIG))
    validate.add_argument("--events", default=str(DEFAULT_EVENTS))
    validate.add_argument("--features", default=str(DEFAULT_FEATURES))
    validate.add_argument("--output-dir", default="outputs/backtesting/v2.4-local")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-mock":
        config = WeevillTrakConfig(str(DEFAULT_CONFIG))
        events = add_configured_manual_events(load_events(args.events), config)
        path = generate_mock_engineered_features(events, args.output)
        print(path)
    elif args.command == "train":
        manifest = train_local(args.config, args.events, args.features, args.artifacts)
        print(json.dumps(manifest, indent=2))
    elif args.command == "predict":
        result = predict_local(
            args.date, args.features, args.artifacts, args.events, args.output
        )
        print(result.to_string(index=False))
    elif args.command == "walk-forward":
        _, summary = walk_forward_local(
            args.config, args.events, args.features, args.output_dir
        )
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
    add_configured_manual_events,

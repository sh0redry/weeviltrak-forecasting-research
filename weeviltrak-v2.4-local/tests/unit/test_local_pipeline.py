import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app.config.weevilltrak_config import WeevillTrakConfig
from app.local.pipeline import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    build_training_matrix,
    generate_mock_engineered_features,
    load_engineered_features,
)

TEST_TMP = PROJECT_ROOT / "outputs/test_tmp"
TEST_TMP.mkdir(parents=True, exist_ok=True)


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "location_id": ["a", "a", "a", "a"],
            "latitude": [40.0] * 4,
            "longitude": [-75.0] * 4,
            "stage_id": [1, 4, 2, 3],
            "stage_date": pd.to_datetime(
                ["2025-04-01", "2025-04-10", "2025-04-20", "2025-05-05"]
            ),
            "event_year": [2025] * 4,
        }
    )


class LocalPipelineTests(unittest.TestCase):
    def test_mock_features_are_deterministic(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            root = Path(directory)
            first = generate_mock_engineered_features(_events(), root / "first.csv")
            second = generate_mock_engineered_features(_events(), root / "second.csv")
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertTrue(first.with_suffix(".csv.metadata.json").exists())

    def test_signed_target_includes_post_event_rows(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            config = WeevillTrakConfig(str(DEFAULT_CONFIG))
            path = generate_mock_engineered_features(
                _events(), Path(directory) / "features.csv"
            )
            daily = load_engineered_features(path, config.features)
            x_train, y_train = build_training_matrix(
                _events(), daily, config, cutoff_date="2026-01-01"
            )
            self.assertEqual(list(x_train.columns), config.features)
            self.assertEqual(y_train[config.target].min(), -10)
            self.assertGreater(y_train[config.target].max(), 0)

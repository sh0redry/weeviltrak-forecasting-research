import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from griddedweather.s3_store import S3Manager

TEST_TMP = Path(__file__).resolve().parents[2] / "outputs/test_tmp"
TEST_TMP.mkdir(parents=True, exist_ok=True)


class LocalStorageTests(unittest.TestCase):
    def test_local_store_round_trip(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory, patch.dict(
            os.environ, {"WEEVILTRAK_LOCAL_STORE": directory}
        ):
            store = S3Manager("local")
            expected = pd.DataFrame({"value": [1, 2]})
            store.upload_file(expected, "nested/sample.csv")
            pd.testing.assert_frame_equal(store.read_file("nested/sample.csv"), expected)

    def test_local_store_rejects_path_escape(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory, patch.dict(
            os.environ, {"WEEVILTRAK_LOCAL_STORE": directory}
        ):
            store = S3Manager("local")
            with self.assertRaises(ValueError):
                store.upload_file("bad", "../escape.txt")

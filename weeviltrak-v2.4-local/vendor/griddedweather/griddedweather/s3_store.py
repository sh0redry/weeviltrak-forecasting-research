"""Filesystem-backed stand-in for the private S3 manager."""

from __future__ import annotations

import io
import os
import pickle
from pathlib import Path

import pandas as pd


class S3Manager:
    def __init__(self, bucket_name: str = "local"):
        self.bucket_name = bucket_name
        self.root = Path(os.getenv("WEEVILTRAK_LOCAL_STORE", "outputs/local/object_store"))
        self.client = None

    def _path(self, key: str) -> Path:
        clean = str(key).replace("\\", "/").lstrip("/")
        prefix = f"{self.bucket_name}/"
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
        path = (self.root / clean).resolve()
        root = self.root.resolve()
        if root not in path.parents and path != root:
            raise ValueError(f"Object key escapes local store: {key}")
        return path

    def upload_file(self, value, key: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, pd.DataFrame):
            if path.suffix == ".parquet":
                value.to_parquet(path, index=False)
            else:
                value.to_csv(path, index=False)
        elif isinstance(value, str):
            path.write_text(value, encoding="utf-8")
        elif isinstance(value, (bytes, bytearray)):
            path.write_bytes(value)
        else:
            with path.open("wb") as handle:
                pickle.dump(value, handle)

    def read_file(self, key: str):
        path = self._path(key)
        if not path.exists():
            raise FileNotFoundError(f"Local object does not exist: {path}")
        if path.suffix == ".csv":
            return pd.read_csv(path)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix in {".yml", ".yaml", ".json", ".txt"}:
            return path.read_text(encoding="utf-8")
        with path.open("rb") as handle:
            return pickle.load(handle)

    def file_exists(self, key: str) -> bool:
        return self._path(key).exists()

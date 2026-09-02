"""Small offline-compatible subset of the private ``griddedweather`` API.

It intentionally does not emulate Redshift weather queries.  The local research
CLI consumes an explicit engineered-feature file; production users must install
the approved private package instead of this compatibility layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import pandas as pd


@dataclass
class RedshiftCredentials:
    host: str = ""
    port: int = 5439
    database: str = ""
    user: str = ""
    password: str = ""


@dataclass
class WeatherConfig:
    redshift: RedshiftCredentials = field(default_factory=RedshiftCredentials)
    s3_bucket: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "WeatherConfig":
        return cls(
            redshift=RedshiftCredentials(
                host=env.get("DATABASE_HOST", env.get("REDSHIFT_HOST", "")),
                port=int(env.get("DATABASE_PORT", env.get("REDSHIFT_PORT", "5439"))),
                database=env.get("DATABASE_NAME", env.get("REDSHIFT_DATABASE", "")),
                user=env.get("REDSHIFT_USER", ""),
                password=env.get("REDSHIFT_PASSWORD", ""),
            ),
            s3_bucket=env.get("S3_BUCKET_NAME", ""),
        )


class GriddedWeatherClient:
    """Fail clearly when a production weather pull is attempted offline."""

    def __init__(self, config: WeatherConfig):
        self.config = config

    def pull_weather_data(self, _requests):
        raise RuntimeError(
            "Live weather pulls are unavailable in local mode. Export engineered "
            "daily weather features on an authorized machine and pass them to "
            "`python -m app.local.cli`, or use `prepare-mock` for a smoke test."
        )


from .s3_store import S3Manager

__all__ = [
    "GriddedWeatherClient",
    "RedshiftCredentials",
    "S3Manager",
    "WeatherConfig",
]

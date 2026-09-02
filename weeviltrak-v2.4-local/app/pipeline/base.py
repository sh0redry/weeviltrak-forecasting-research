"""
Shared base class for WeevilTrak pipeline orchestrators.

Extracts the identical service initialisation (config, DB, S3, data-prep,
model-manager) that was duplicated between WeevilTrakPipeline and
BacktestingFramework.
"""

from app.config.weevilltrak_config import WeevillTrakConfig
from app.models.factory import create_model_manager
from app.services.data_preparation_service import DataPreparationService
from app.services.database_service import DatabaseManager
from griddedweather.s3_store import S3Manager
from app.settings import setup_logger

logger = setup_logger()


class PipelineBase:
    """
    Common initialisation for all WeevilTrak pipelines.

    Subclasses get ``self.config``, ``self.db_manager``, ``self.s3_manager``,
    ``self.data_service``, and ``self.model_manager`` for free.
    """

    def __init__(self, config_path: str):
        self.config = WeevillTrakConfig(config_path)

        bucket_name = self.config.config.get("data_source", {}).get(
            "s3_bucket", "sps-ds-bucket"
        )

        self.db_manager = DatabaseManager(self.config)
        self.s3_manager = S3Manager(bucket_name)
        self.data_service = DataPreparationService(
            self.db_manager,
            self.s3_manager,
            s3_bucket=bucket_name,
        )
        self.model_manager = create_model_manager(self.config, self.s3_manager)

    def resolve_training_window_years(self, override=None):
        """Return an explicit window or the model configuration default."""
        value = (
            override
            if override is not None
            else self.config.config.get("training_window_years")
        )
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                "training_window_years must be a positive int or None, "
                f"got {value!r}"
            )
        return value

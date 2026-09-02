"""
Factory for creating model managers based on YAML config.
"""

from app.config.weevilltrak_config import WeevillTrakConfig
from app.models.base import BaseModelManager
from griddedweather.s3_store import S3Manager


def create_model_manager(
    config: WeevillTrakConfig, s3_manager: S3Manager
) -> BaseModelManager:
    """
    Instantiate the correct model manager based on ``model_type`` in config.

    Currently supported:
        - ``random_forest`` (default) → :class:`RFModelManager`
        - ``lightgbm`` → :class:`LGBMModelManager`
        - ``lightgbm_gp`` → temporary alias to :class:`LGBMModelManager`
          during the monotonic-constraints-first implementation slice
    """
    model_type = config.config.get("model_type", "random_forest")

    if model_type == "random_forest":
        from app.models.model_manager import RFModelManager

        return RFModelManager(config, s3_manager)
    if model_type in {"lightgbm", "lightgbm_gp"}:
        from app.models.model_manager import LGBMModelManager

        return LGBMModelManager(config, s3_manager)
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}' in config. "
            f"Supported: random_forest, lightgbm, lightgbm_gp"
        )

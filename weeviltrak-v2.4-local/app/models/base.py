"""
Abstract base class for WeevilTrak model managers.

All model implementations (RandomForest, LightGBM+GP, etc.) must implement
this interface so that pipeline code works uniformly regardless of model type.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from app.config.weevilltrak_config import WeevillTrakConfig
from griddedweather.s3_store import S3Manager


class BaseModelManager(ABC):
    """
    Interface every model manager must implement.

    Pipeline code calls these four methods exclusively — no direct access
    to the underlying sklearn / LightGBM / GP objects.
    """

    def __init__(self, config: WeevillTrakConfig, s3_manager: S3Manager):
        self.config = config
        self.s3_manager = s3_manager
        self.model = None

    @abstractmethod
    def train(self, x_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs):
        """Fit the model on training data. Stores the fitted object on ``self.model``."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return a 1-D array of predictions (same length as *X*)."""

    @abstractmethod
    def save_model(self, model=None, model_path: Optional[str] = None) -> None:
        """Persist model to S3 (or local path)."""

    @abstractmethod
    def load_model(self, model_path: Optional[str] = None):
        """Load a previously saved model. Stores on ``self.model`` and returns it."""

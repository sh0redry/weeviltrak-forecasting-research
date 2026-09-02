"""
Model management services for training and prediction.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from app.config.weevilltrak_config import WeevillTrakConfig
from app.models.base import BaseModelManager
from app.settings import setup_logger
from griddedweather.s3_store import S3Manager

logger = setup_logger()

VALID_MONOTONE_CONSTRAINT_VALUES = {-1, 0, 1}


def build_monotone_constraints(
    features: List[str],
    constraints_by_feature: Optional[Dict[str, int]],
) -> Optional[List[int]]:
    """
    Convert feature-name monotonic constraints to LightGBM's ordered list format.
    """
    if not constraints_by_feature:
        return None

    unknown_features = sorted(set(constraints_by_feature) - set(features))
    if unknown_features:
        raise ValueError(
            "Unknown monotone constraint feature(s): "
            f"{unknown_features}. Valid features: {features}"
        )

    invalid_values = {
        feature: value
        for feature, value in constraints_by_feature.items()
        if value not in VALID_MONOTONE_CONSTRAINT_VALUES
    }
    if invalid_values:
        raise ValueError(
            "Invalid monotone constraint value(s): "
            f"{invalid_values}. Allowed values: {-1, 0, 1}"
        )

    constraints = [int(constraints_by_feature.get(feature, 0)) for feature in features]
    if len(constraints) != len(features):
        raise ValueError(
            "Generated monotone_constraints length does not match features length: "
            f"{len(constraints)} != {len(features)}"
        )

    return constraints


class RFModelManager(BaseModelManager):
    """
    RandomForest model manager implementing BaseModelManager.
    """

    def __init__(self, config: WeevillTrakConfig, s3_manager: S3Manager):
        super().__init__(config, s3_manager)
        logger.info("RFModelManager initialized")

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        **kwargs,
    ) -> RandomForestRegressor:
        logger.info(f"Training RF model with {len(x_train)} samples...")

        model_params = self.config.model_params.copy()
        model_params.update(kwargs)

        if "monotone_constraints_by_feature" in model_params:
            logger.info(
                "Ignoring monotone_constraints_by_feature for random_forest model_type"
            )
            model_params.pop("monotone_constraints_by_feature", None)

        self.model = RandomForestRegressor(**model_params)
        self.model.fit(x_train, y_train.values.ravel())

        logger.info("RF model training completed successfully")
        return self.model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("No model loaded — call train() or load_model() first.")
        return self.model.predict(X)

    def predict_with_intervals(
        self,
        x_test: pd.DataFrame,
        y_train: pd.DataFrame,
        alpha: Optional[float] = None,
    ) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("No model loaded — call train() or load_model() first.")

        logger.info("Making RF predictions with intervals...")
        y_pred = self.model.predict(x_test)
        pred_dict = {"pred": y_pred}

        if alpha is not None:
            if (
                not hasattr(self.model, "oob_prediction_")
                or self.model.oob_prediction_ is None
            ):
                logger.warning(
                    "Model does not have OOB predictions. "
                    "Confidence intervals require oob_score=True during training."
                )
                raise ValueError(
                    "Cannot calculate confidence intervals: "
                    "model was not trained with oob_score=True"
                )

            error_oob = self.model.oob_prediction_ - y_train.values.ravel()
            quant_oob = np.quantile(np.abs(error_oob), 1 - alpha)
            pred_dict["pred_lower"] = y_pred - quant_oob
            pred_dict["pred_upper"] = y_pred + quant_oob
        else:
            logger.info("No alpha provided, no confidence intervals will be calculated.")

        pred = pd.DataFrame(pred_dict, index=x_test.index)

        columns_to_include = ["stage_id", "latitude", "longitude"]
        if "location_id" in x_test.columns:
            columns_to_include.append("location_id")

        pred_data = pd.concat([x_test[columns_to_include], pred], axis=1)
        logger.info(
            f"Prediction completed for {x_test.index.min().strftime('%Y-%m-%d')} "
            f"to {x_test.index.max().strftime('%Y-%m-%d')}"
        )
        return pred_data

    def save_model(self, model=None, model_path: str = None) -> None:
        if model_path is None:
            model_path = self.config.config["model_path"]
        if model is None:
            model = self.model
        self.s3_manager.upload_file(model, model_path)
        logger.info("Model saved successfully")

    def load_model(self, model_path: str = None) -> RandomForestRegressor:
        if model_path is None:
            model_path = self.config.config["model_path"]

        logger.info(f"Loading model from {model_path}...")
        self.model = self.s3_manager.read_file(model_path)
        logger.info("Model loaded successfully")
        return self.model


class LGBMModelManager(BaseModelManager):
    """
    LightGBM model manager with optional monotonic constraints.
    """

    def __init__(self, config: WeevillTrakConfig, s3_manager: S3Manager):
        super().__init__(config, s3_manager)
        logger.info("LGBMModelManager initialized")

    def _coerce_feature_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        LightGBM requires numeric/bool pandas dtypes.
        """
        if not isinstance(X, pd.DataFrame):
            return X

        coerced = X.copy()
        for col in coerced.columns:
            if not (
                pd.api.types.is_bool_dtype(coerced[col])
                or pd.api.types.is_numeric_dtype(coerced[col])
            ):
                coerced[col] = pd.to_numeric(coerced[col], errors="coerce")
        return coerced

    def _build_lgbm_params(self, **kwargs) -> Dict[str, object]:
        model_params = self.config.model_params.copy()
        model_params.update(kwargs)

        constraints_by_feature = model_params.pop(
            "monotone_constraints_by_feature", None
        )
        monotone_constraints = build_monotone_constraints(
            self.config.features,
            constraints_by_feature,
        )
        if monotone_constraints is not None:
            model_params["monotone_constraints"] = monotone_constraints
            logger.info(
                "Applied LightGBM monotone constraints: %s",
                {
                    feature: constraint
                    for feature, constraint in zip(
                        self.config.features, monotone_constraints
                    )
                },
            )

        return model_params

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        **kwargs,
    ):
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError(
                "lightgbm is required for model_type 'lightgbm' but is not installed."
            ) from exc

        logger.info(f"Training LightGBM model with {len(x_train)} samples...")
        model_params = self._build_lgbm_params(**kwargs)
        x_train = self._coerce_feature_frame(x_train)

        self.model = LGBMRegressor(**model_params)
        self.model.fit(x_train, y_train.values.ravel())

        logger.info("LightGBM model training completed successfully")
        return self.model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("No model loaded — call train() or load_model() first.")
        X = self._coerce_feature_frame(X)
        return self.model.predict(X)

    def save_model(self, model=None, model_path: str = None) -> None:
        if model_path is None:
            model_path = self.config.config["model_path"]
        if model is None:
            model = self.model
        self.s3_manager.upload_file(model, model_path)
        logger.info("Model saved successfully")

    def load_model(self, model_path: str = None):
        if model_path is None:
            model_path = self.config.config["model_path"]

        logger.info(f"Loading model from {model_path}...")
        self.model = self.s3_manager.read_file(model_path)
        logger.info("Model loaded successfully")
        return self.model


# Backward-compatible alias for existing imports.
ModelManager = RFModelManager

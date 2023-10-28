from pyexpat import features
from typing import Union, Callable

import torch


import numpy as np
import pytorch_lightning as pl
from sklearn.base import ClassifierMixin, RegressorMixin
from torchmetrics.metric import Metric

from fpl_ml.log import log_metrics_and_visualisations

class SklearnModel:
    """Wrapper for sklearn models for standardised logging."""

    def __init__(
        self,
        model: Union[ClassifierMixin, RegressorMixin],
        metrics: dict[str, Metric],
        visualisations: dict[str, Callable],
        features_key: str = 'X',
        labels_key: str = 'y'
    ):
        self.model = model
        self._metrics = metrics
        self._visualisations = visualisations

        self._features_key = features_key
        self._labels_key = labels_key

    def __call__(self, batch):
        if self.model._estimator_type == "regressor":
            return self.model.predict(batch)
        elif self.model._estimator_type == "classifier":
            return self.model.predict_proba(batch)

    def fit(self, data_module: pl.LightningDataModule) -> dict[str, float]:
        """Fits sklearn model and logs metrics and visualisations on the training
        and validation data.

        Args:
            data_module: Datamodule containing training and validation data

        Returns:
            Dictionary of metric names and their values
        """

        # Fit model using whole train dataset
        train_batch = data_module.train_dataloader().dataset[:]
        self.model.fit(X=train_batch[self._features_key], y=train_batch[self._labels_key])

        # Log metrics and figures for train data
        output_metrics = self._log(batch=train_batch, stage="train")

        # Log metrics and figures for validation data
        val_batch = data_module.val_dataloader().dataset[:]
        output_metrics = self._log(batch=val_batch, stage="val")

        return output_metrics

    def _log(self, batch: dict[str, np.ndarray], stage: str):
        y_true = batch[self._labels_key]

        if self.model._estimator_type == "regressor":
            y_pred = self.model.predict(batch[self._features_key])

            # Reshape output of prediction for torchmetrics compatibility when we have
            # a single output
            if y_true.shape[1] == 1:
                y_pred = y_pred.reshape(-1, 1)

        else:
            raise NotImplementedError(
                f"Metric logging for estimator type: {self.model._estimator_type} not yet implemented"
            )

        # Move to pytorch tensor for torchmetrics compatibility
        y_true = torch.from_numpy(y_true)
        y_pred = torch.from_numpy(y_pred)

        output_metrics = log_metrics_and_visualisations(
            y_true,
            y_pred,
            stage=stage,
            metrics=self._metrics,
            visualisations=self._visualisations,
        )

        return output_metrics

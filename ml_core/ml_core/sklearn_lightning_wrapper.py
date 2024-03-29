from typing import Callable, Union

import numpy as np
import lightning.pytorch as pl
import torch
from sklearn.base import ClassifierMixin, RegressorMixin
from torchmetrics.metric import Metric

from ml_core.log import log_metrics_and_visualisations


class SklearnModel:
    """Wrapper for sklearn models for standardised logging."""

    def __init__(
        self,
        model: Union[ClassifierMixin, RegressorMixin],
        metrics: dict[str, Metric],
        visualisations: dict[str, Callable],
        features_key: str = "X",
        labels_key: str = "y",
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
        self.model.fit(
            X=train_batch[self._features_key], y=train_batch[self._labels_key]
        )

        # Log metrics and figures for train data
        train_output_metrics = self._log(batch=train_batch, stage="train")

        # Log metrics and figures for validation data
        val_batch = data_module.val_dataloader().dataset[:]
        val_output_metrics = self._log(batch=val_batch, stage="val")

        # Log metrics and figures for test data
        data_module.setup("test")
        test_batch = data_module.test_dataloader().dataset[:]
        test_output_metrics = self._log(batch=test_batch, stage="test")

        output_metrics = {}
        output_metrics.update(train_output_metrics)
        output_metrics.update(val_output_metrics)
        output_metrics.update(test_output_metrics)

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
        y_true = torch.from_numpy(y_true).float().to("cpu", dtype=torch.float32)
        y_pred = torch.from_numpy(y_pred).float().to("cpu", dtype=torch.float32)

        output_metrics = log_metrics_and_visualisations(
            y_true,
            y_pred,
            stage=stage,
            metrics=self._metrics,
            visualisations=self._visualisations,
        )

        return output_metrics

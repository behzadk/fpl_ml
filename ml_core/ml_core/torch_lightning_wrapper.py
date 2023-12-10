import lightning.pytorch as pl
import torch
from loguru import logger
from ml_core.log import log_metrics_and_visualisations
from functools import partial
import torchmetrics
from lightning import LightningModule

class BasicTorchModel(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        visualisations,
        metrics: dict[str, torchmetrics.Metric],
        features_key: str = "X",
        labels_key: str = "y",
    ):
        super().__init__()

        self._model = model
        self._features_key = features_key
        self._labels_key = labels_key
        self._visualisations = visualisations
        self._metrics = metrics


    def forward(self, inputs, target):
        return self._call_model(inputs, target)
    
    def setup_train(
        self,
        data_module: pl.LightningDataModule,
        metrics: dict[str, torchmetrics.Metric],
        partial_optimizer:partial[torch.optim.Optimizer],
        loss: torchmetrics.Metric,
    ):
        self._data_module = data_module

        self._metrics = metrics
        self._set_metric_attributes()


        self._partial_optimizer = partial_optimizer
        self._loss = loss

        self._loss.is_differentiable = True

    def training_step(self, batch, batch_idx):
        y_hat = self._call_model(batch)
        loss = self._call_batch_loss(y_hat, batch)
        self.log("loss", loss, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self._call_model(batch)
        loss = self._call_batch_loss(y_hat, batch)

        self.log("val_loss", loss, on_epoch=True)
        for m in self._metrics:
            getattr(self, f"val_{m}").update(y_hat, batch[self._labels_key])

        return loss


    def on_validation_epoch_end(self):
        for m in self._metrics:
            self.log(f"val_{m}", getattr(self, f"val_{m}").compute())
            getattr(self, f"val_{m}").reset()

    def test_step(self, batch, batch_idx):
        y_hat = self._call_model(batch)
        loss = self._call_batch_loss(y_hat, batch)

        self.log("test_loss", loss, on_epoch=True)

        return loss

    def configure_optimizers(self) -> torch.optim:
        optimizer = self._partial_optimizer(params=self._model.parameters())

        return optimizer

    def _call_batch_loss(self, y_hat, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._loss(y_hat, batch[self._labels_key])

    def _log_metrics(self, batch, stage, log_visualisations=False, log_metrics=True):

        logger.info("Generating validation metrics...")

        metrics = None
        visualisations = None

        if log_visualisations:
            visualisations = self._visualisations

        if log_metrics:
            metrics = self._metrics

        eval_metrics = log_metrics_and_visualisations(
            y_true=batch[self._labels_key],
            y_pred=self._call_model(batch),
            stage=stage,
            metrics=metrics,
            visualisations=visualisations,
        )

        return eval_metrics

    def _call_model(self, batch: dict[str, torch.tensor]) -> torch.Tensor:
        return self._model(batch[self._features_key])
    

    def _set_metric_attributes(self):
        """Sets metrics as attributes using the key from the
        dictionary as the name.

        This allows for logging of values over time
        """

        for key in self._metrics:
            setattr(self, f"train_{key}", self._metrics[key].clone())
            setattr(self, f"val_{key}", self._metrics[key].clone())
            setattr(self, f"test_{key}", self._metrics[key].clone())
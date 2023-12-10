from typing import Callable, Optional

import mlflow
import torch
from torchmetrics import Metric
from loguru import logger

def log_metrics_and_visualisations(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    stage: str,
    metrics: Optional[dict[str, Metric]],
    visualisations: Optional[dict[str, Callable]],
) -> dict[str, float]:
    """Standardised logging of metrics and visualisations with mlflow

    Args:
        y_true: True data.
        y_pred: Predicted data from model.
        stage: Stage of workflow (train, val, test).
        metrics: Dictionary of torchmetrics.
        visualisations: Dictionary of visualisation functions.

    Returns:
        Dictionary of metrics names and their values
    """

    eval_metrics = {}

    if metrics:
        # Iterate, compute and log validation metrics
        for m in metrics.keys():
            met = metrics[m].update(y_pred, y_true)
            mlflow.log_metric(f"{stage}_{m}", metrics[m].compute())

            eval_metrics[f"{stage}_{m}"] = met

            logger.info(f"{stage}_{m}")


    if visualisations:
        # Apply visualisation steps to validation data
        for vis_key in visualisations.keys():
            fig = visualisations[vis_key](y_true=y_true, y_pred=y_pred)
            mlflow.log_figure(fig, f"{stage}_{vis_key}.html")

    return eval_metrics

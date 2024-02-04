import time
from typing import Optional, cast, Union

import pandas as pd
from lightning.pytorch import seed_everything
import mlflow
import shutil
from sklearn.base import ClassifierMixin, RegressorMixin
import torch

def get_experiment_runs(
    experiment_name: str,
    tracking_uri: str,
) -> pd.DataFrame:
    """Gets all runs of an experiment.
    Args:
        experiment_name (str): Name of MLflow experiment.
        tracking_uri (str): The URI for the MLflow tracking server.
    Returns:
        Dataframe of MLflow runs.
    """

    mlflow.set_tracking_uri(tracking_uri)

    current_experiment = mlflow.get_experiment_by_name(experiment_name)

    runs_df = mlflow.search_runs([current_experiment.experiment_id])

    return runs_df

def _load_sklearn_model(run_id: str) -> Union[ClassifierMixin, RegressorMixin]:
    """Loads a scikit-learn model from the specified MLflow run.

    Args:
        run_id (str): ID of the MLflow run containing the model.

    Returns:
        object: The loaded scikit-learn model.
    """
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")


def _load_torch_model(run_id: str) -> torch.nn.Module:
    """Loads a PyTorch model from the specified MLflow run.

    Args:
        run_id (str): ID of the MLflow run containing the model.

    Returns:
        object: The loaded PyTorch model.
    """
    return mlflow.pytorch.load_model(f"runs:/{run_id}/model")


def load_model(run_id: str, tracking_uri: str) -> Union[torch.nn.Module, ClassifierMixin, RegressorMixin]:
    """Loads a model from the specified MLflow run based on available flavors.

    Args:
        run_id (str): ID of the MLflow run containing the model.
        tracking_uri (str): The URI for the MLflow tracking server.

    Returns:
        object: The loaded model (either scikit-learn or PyTorch).
    """
    mlflow.set_tracking_uri(tracking_uri)

    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

    if "pytorch" in model.metadata.flavors.keys():
        model = _load_torch_model(run_id)

    elif "sklearn" in model.metadata.flavors.keys():
        model = _load_sklearn_model(run_id)

    else:
        raise NotImplementedError(
            f"Model flavor {model.metadata.flavors.keys()} and not 'pytorch' or 'sklearn' "
        )

    return model


def load_preprocessing_pipeline(run_id, tracking_uri):
    """Loads preprocessing pipeline from the specified MLflow run.

    Args:
        run_id (str): ID of the MLflow run containing the model.
        tracking_uri (str): The URI for the MLflow tracking server.

    Returns:
        _description_
    """
    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/preprocessing_pipeline/").unwrap_python_model()
    return model

from enum import Enum
import os
from typing import Optional
import hydra_zen as hz

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from torchmetrics import MeanSquaredError, SpearmanCorrCoef, R2Score
from sklearn.svm import LinearSVR

from ml_core.data_module import DataModuleLoadedFromCSV
from ml_core.dataset import Dataset
from ml_core.preprocessing import (
    DataframePipeline,
    RandomSplitData,
    SplitFeaturesAndLabels,
)
from ml_core.user import User
from ml_core.visualisations import plot_regression_scatter
from ml_core.torch_models import NeuralNetwork
from torch import nn
import torch
import lightning.pytorch as pl


class StoreGroups(Enum):
    USER = "user"
    DATASET = "dataset"
    DATAMODULE = "datamodule"
    MODEL = "model"
    PREPROCESSING = "preprocessing"
    DATA_SPLITTER = "data_splitter"
    METRICS = "metrics"
    VISUALISATIONS = "visualisations"

    # Pytorch specific
    TORCH_MODEL = "torch_model"
    OPTIMIZER = "optimizer"
    TRAINER = "trainer"
    CALLBACKS = "callbacks"
    LOSS = "loss"


class HydraGroups(Enum):
    HYDRA_SWEEPER = "hydra/sweeper"
    HYDRA_SWEEPER_SAMPLER = "hydra/sweeper/sampler"


def _initialize_users(mlruns_dir: os.PathLike):
    user_store = hz.store(group=StoreGroups.USER.value)
    user_store(User, mlruns_dir=mlruns_dir, device="cpu", name="default-cpu")
    user_store(User, mlruns_dir=mlruns_dir, device="gpu", name="default-gpu")


def _initialize_datasets():
    dataset_store = hz.store(group=StoreGroups.DATASET.value)
    dataset_store(Dataset, device="${user.device}", zen_partial=True, name="default")


def _initialize_datamodules(
    train_val_data_path: Optional[os.PathLike] = None,
    test_data_path: Optional[os.PathLike] = None,
    predict_data_path: Optional[os.PathLike] = None,
):
    datamodule_store = hz.store(group=StoreGroups.DATAMODULE.value)
    datamodule_store(
        DataModuleLoadedFromCSV,
        train_validation_data_path=train_val_data_path,
        test_data_path=test_data_path,
        predict_data_path=predict_data_path,
        partial_dataset="${dataset}",
        preprocessing_pipeline="${preprocessing}",
        train_validation_splitter="${data_splitter}",
        precision=32,
        use_torch=False,
        zen_partial=True,
        name="default-sklearn",
    )

    datamodule_store(
        DataModuleLoadedFromCSV,
        train_validation_data_path=train_val_data_path,
        test_data_path=test_data_path,
        predict_data_path=predict_data_path,
        partial_dataset="${dataset}",
        preprocessing_pipeline="${preprocessing}",
        train_validation_splitter="${data_splitter}",
        batch_size=32,
        precision=32,
        use_torch=True,
        zen_partial=True,
        name="default-torch",
    )


def _initialize_models():
    model_store = hz.store(group=StoreGroups.MODEL.value)

    sklearn_regressors = [RandomForestRegressor, GradientBoostingRegressor, LinearSVR]

    # Sklearn regressors
    for regressor in sklearn_regressors:
        model_store(regressor, name=regressor.__name__)

    model_store(
        NeuralNetwork,
        input_dim=48,
        output_dim=1,
        hidden_layer_sizes=[64],
        activation_function=nn.ReLU,
        dropout=0.2,
        output_function=None,
        name="neural_network",
    )


def _initialize_model_wrappers():
    """TODO; add model wrappers for sklearn models and torch model"""
    pass


def _initialize_metrics():
    metrics_store = hz.store(group=StoreGroups.METRICS.value)

    mse = hz.builds(MeanSquaredError, squared=True, num_outputs=1, hydra_convert="all")
    mae = hz.builds(MeanSquaredError, squared=False, num_outputs=1, hydra_convert="all")
    spearman_rank = hz.builds(SpearmanCorrCoef, num_outputs=1, hydra_convert="all")
    r2_score = hz.builds(R2Score, num_outputs=1, hydra_convert="all")

    regression_metrics = {
        "MSE": mse,
        "MAE": mae,
        "SpearmanCorrCoef": spearman_rank,
        "R2Score": r2_score,
    }

    metrics_store(regression_metrics, name="regression_default", hydra_convert="all")


def _initialize_preprocessing():
    preprocessing_store = hz.store(group=StoreGroups.PREPROCESSING.value)

    split_step = hz.builds(
        SplitFeaturesAndLabels,
        x_column_prefixes=["X_"],
        y_columns=["total_points"],
        hydra_convert="all",
    )

    steps = [split_step]

    preprocessing_store(
        DataframePipeline, dataframe_processing_steps=steps, name="default"
    )


def _initialize_data_splitters():
    data_splitter_store = hz.store(group=StoreGroups.DATA_SPLITTER.value)

    data_splitter_store(RandomSplitData, frac=0.8, name="default")


def _initialize_visualisations():
    visualisations_store = hz.store(group=StoreGroups.VISUALISATIONS.value)

    regression_scatter = hz.builds(
        plot_regression_scatter, trendline=None, identity=True, zen_partial=True
    )
    regression_default = {"scatter": regression_scatter}

    visualisations_store(regression_default, name="regression_default")


def _initialize_optimizers():
    optimizer_store = hz.store(group=StoreGroups.OPTIMIZER.value)
    optimizer_store(
        torch.optim.Adam, lr=0.001, weight_decay=0.0, name="default", zen_partial=True
    )


def _initialize_trainers():
    trainer_store = hz.store(group=StoreGroups.TRAINER.value)
    trainer_store(
        pl.Trainer,
        max_epochs=1,
        accelerator="${user.device}",
        precision="${datamodule.precision}",
        callbacks="${callbacks}",
        name="default",
        log_every_n_steps=1,
    )


def _initialize_callbacks():
    callbacks_store = hz.store(group=StoreGroups.CALLBACKS.value)

    early_stop = hz.builds(
        pl.callbacks.early_stopping.EarlyStopping,
        monitor="val_loss",
        mode="min",
        min_delta=0.001,
        patience=0,
    )

    callbacks_store([early_stop], name="default")


def _initialize_loss():
    loss_store = hz.store(group=StoreGroups.LOSS.value)
    loss_store(nn.MSELoss, reduction="mean", name="MSE")
    loss_store(nn.L1Loss, reduction="mean", name="MAE")


def initialize_stores(
    mlruns_dir,
    train_val_data_path: Optional[os.PathLike] = None,
    test_data_path: Optional[os.PathLike] = None,
    predict_data_path: Optional[os.PathLike] = None,
):
    _initialize_users(mlruns_dir=mlruns_dir)
    _initialize_datasets()
    _initialize_datamodules(
        train_val_data_path=train_val_data_path,
        test_data_path=test_data_path,
        predict_data_path=predict_data_path,
    )
    _initialize_models()
    _initialize_preprocessing()
    _initialize_data_splitters()
    _initialize_metrics()
    _initialize_visualisations()

    # Pytorch specific
    _initialize_optimizers()
    _initialize_trainers()
    _initialize_callbacks()
    _initialize_loss()

    hz.store.add_to_hydra_store(overwrite_ok=False)

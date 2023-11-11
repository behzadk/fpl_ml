from enum import Enum
import os
from typing import Optional
import hydra_zen as hz

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from torchmetrics import MeanSquaredError

from ml_core.data_module import DataModuleLoadedFromCSV
from ml_core.dataset import Dataset
from ml_core.preprocessing import (
    DataframePipeline,
    RandomSplitData,
    SplitFeaturesAndLabels,
)
from ml_core.user import User
from ml_core.visualisations import plot_regression_scatter


class StoreGroups(Enum):
    USER = "user"
    DATASET = "dataset"
    DATAMODULE = "datamodule"
    MODEL = "model"
    PREPROCESSING = "preprocessing"
    DATA_SPLITTER = "data_splitter"
    METRICS = "metrics"
    VISUALISATIONS = "visualisations"

class HydraGroups(Enum):
    HYDRA_SWEEPER = "hydra/sweeper"
    HYDRA_SWEEPER_SAMPLER = "hydra/sweeper/sampler"


def _initialize_users(mlruns_dir: os.PathLike):
    user_store = hz.store(group=StoreGroups.USER.value)
    user_store(User, mlruns_dir=mlruns_dir, device="cpu", name="default")
    user_store(User, mlruns_dir=mlruns_dir, device="gpu", name="gpu")


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
        zen_partial=True,
        name="default",
    )


def _initialize_models():
    model_store = hz.store(group=StoreGroups.MODEL.value)

    sklearn_regressors = [RandomForestRegressor, GradientBoostingRegressor]

    # Sklearn regressors
    for regressor in sklearn_regressors:
        model_store(regressor, name=regressor.__name__)


def _initialize_metrics():
    metrics_store = hz.store(group=StoreGroups.METRICS.value)

    mse = hz.builds(MeanSquaredError, squared=True, num_outputs=1, hydra_convert="all")
    regression_metrics = {"MSE": mse}
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

    hz.store.add_to_hydra_store(overwrite_ok=False)


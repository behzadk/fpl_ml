import os
from dataclasses import dataclass
from typing import Any, Optional

from hydra_zen import MISSING, ZenField, builds, make_config, store
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from torchmetrics import MeanSquaredError

from fpl_ml.data_module import DataModuleLoadedFromCSV
from fpl_ml.dataset import Dataset
from fpl_ml.preprocessing import (
    DataframePipeline,
    OneHotEncodeColumns,
    RandomSplitData,
    SplitFeaturesAndLabels,
    StandardScaleColumns,
)
from fpl_ml.user import User
from fpl_ml.visualisations import plot_regression_scatter


@dataclass
class BaseConfig:
    # Must be passed at command line -- neccesary arguments
    user: Any = MISSING
    dataset: Any = MISSING
    datamodule: Any = MISSING
    model: Any = MISSING
    preprocessing: Any = MISSING
    data_splitter: Any = MISSING
    metrics: Any = MISSING
    visualisations: Any = MISSING


def _initialize_users(mlruns_dir: os.PathLike):
    user_store = store(group="user")
    user_store(User, mlruns_dir=mlruns_dir, device="cpu", name="default")
    user_store(User, mlruns_dir=mlruns_dir, device="gpu", name="gpu")


def _initialize_datasets():
    dataset_store = store(group="dataset")
    dataset_store(Dataset, device="${user.device}", zen_partial=True, name="default")


def _initialize_datamodules(
    train_val_data_path: Optional[os.PathLike] = None,
    test_data_path: Optional[os.PathLike] = None,
    predict_data_path: Optional[os.PathLike] = None,
):
    datamodule_store = store(group="datamodule")
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
    model_store = store(group="model")

    sklearn_regressors = [RandomForestRegressor, GradientBoostingRegressor]

    # Sklearn regressors
    for regressor in sklearn_regressors:
        model_store(regressor, name=regressor.__name__)


def _initialize_metrics():
    metrics_store = store(group="metrics")

    mse = builds(MeanSquaredError, squared=True, num_outputs=1, hydra_convert="all")
    regression_metrics = {"MSE": mse}
    metrics_store(regression_metrics, name="regression_default", hydra_convert="all")


def _initialize_preprocessing():
    preprocessing_store = store(group="preprocessing")

    numerical_features = [
        "X_game_week",
        "X_value",
    ]
    categorical_features = ["X_team", "X_was_home", "X_opponent_team", "X_element_type"]

    steps = []

    scale_step = builds(
        StandardScaleColumns, scale_columns=numerical_features, 
        scale_column_prefixes='X_rolling_', hydra_convert="all"
    )

    encode_step = builds(
        OneHotEncodeColumns, columns_to_encode=categorical_features, hydra_convert="all"
    )

    split_step = builds(
        SplitFeaturesAndLabels,
        x_column_prefixes=["X_"],
        y_columns=["total_points"],
        hydra_convert="all",
    )

    steps = [scale_step, encode_step, split_step]

    preprocessing_store(
        DataframePipeline, dataframe_processing_steps=steps, name="default"
    )


def _initialize_data_splitters():
    data_splitter_store = store(group="data_splitter")

    data_splitter_store(RandomSplitData, frac=0.8, name="default")


def _initialize_visualisations():
    visualisations_store = store(group="visualisations")

    regression_scatter = builds(
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


def collect_config_store():
    zen_config = []

    for value in BaseConfig.__dataclass_fields__.values():
        item = (
            ZenField(name=value.name, hint=value.type, default=value.default)
            if value.default is not MISSING
            else ZenField(name=value.name, hint=value.type)
        )
        zen_config.append(item)

    config = make_config(
        *zen_config,
        hydra_defaults=[
            "_self_",
            dict(user="default"),
            dict(dataset="default"),
            dict(datamodule="default"),
            dict(model="RandomForestRegressor"),
            dict(preprocessing="default"),
            dict(data_splitter="default"),
            dict(metrics="regression_default"),
            dict(visualisations="regression_default"),
        ]
    )

    store(config, group="fpl_ml", name="default_config")

    return store

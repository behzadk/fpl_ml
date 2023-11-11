import os
from functools import partial
from typing import Union

import mlflow
import pytorch_lightning as pl
import torch
import hydra_zen as hz
from hydra_zen import launch, zen
from sklearn.base import RegressorMixin

from fpl_ml.config import collect_config_store
from fpl_ml.stores import initialize_stores
from fpl_ml.preprocessing import DataframePipeline
from fpl_ml.train import train_sklearn
from fpl_ml.user import User
from fpl_ml.utils import set_random_seeds, delete_runs_by_metric

from user_config import PROCESSED_DATA_DIR, MLRUNS_DIR

os.environ["HYDRA_FULL_ERROR"] = "1"


def run_train(
    user: User,
    dataset: partial[torch.utils.data.Dataset],
    datamodule: partial[pl.LightningDataModule],
    model: Union[RegressorMixin, torch.nn.Module],
    preprocessing: DataframePipeline,
    metrics: dict,
    visualisations: dict,
):
    """Main train task function, This takes all the configs and uses them to train an sklearn model.

    TODO:
        Add support for pytorch training

    Args:
        user: Provides information defining mlflow runs dir, device type and random seed.
        dataset: Partially instantiated dataset, used to define how batches are generated.
        datamodule: Partially instantiated data_module, defines how we setup our data and manage dataloaders.
        model: Model we want train.
        preprocessing: Pipeline of `DataframeProcessingStep`'s.
        metrics: Dictionary of metrics to use for qunantifying our model in training and validation.
        visualisations: Dictionary of visualisations we want to apply to our data.
    """

    # Set mlflow tracking uri
    tracking_uri = f"file://{user.mlruns_dir}"
    mlflow.set_tracking_uri(tracking_uri)

    datamodule = datamodule()
    datamodule.setup("fit")

    train_sklearn(
        datamodule,
        model,
        metrics=metrics,
        experiment_name="Test",
        visualisations=visualisations,
        return_validation_metrics=None,
        random_seed=user.random_seed,
    )

    delete_runs_by_metric(mlruns_dir=user.mlruns_dir, experiment_name='Test', keep_n_runs=10, metric='val_MSE', ascending=True)

    # Pseudorandom so mlflow picks a new name for the next run
    set_random_seeds(None)


def train_random_forest_grid_search():
    """Example entry function, performing a grid search across several parameters of the sklearn RandomForestRegressor"""

    config = hz.store.get_entry("fpl_ml", "default_config")
    task_function = zen(run_train)

    # Set parameter values we want for grid search
    overrides = [
        "model=RandomForestRegressor",
        "model.n_estimators=400,500,600,700,800",
        "model.max_features=0.05,0.1,0.25",
        "model.max_depth=8,16",
        "model.min_samples_split=2,4",
    ]

    # Run gridsearch
    (jobs,) = launch(config["node"], task_function, overrides=overrides, multirun=True)

    


def train_gradient_boosting_regressor_grid_search():
    """Example entry function, performing a grid search across several parameters of the sklearn GradientBoostingRegressor"""
    config = hz.store.get_entry("fpl_ml", "default_config")
    task_function = zen(run_train)

    # Set parameter values we want for grid search
    overrides = [
        "model=GradientBoostingRegressor",
        "model.loss=squared_error",
        "model.learning_rate=0.1,0.01,0.001",
        "model.n_estimators=400,500,600,700,800",
        "model.subsample=0.1,0.25,0,4,0.5,0.6",
    ]

    # Run gridsearch
    (jobs,) = launch(config["node"], task_function, overrides=overrides, multirun=True)


def train_single_run():
    """Example entry script, performing a single run using the default config parameters"""
    config = hz.store.get_entry(group="fpl_ml", name="default_config")
    task_function = zen(run_train)

    # Run gridsearch
    launch(config["node"], task_function, multirun=False)



if __name__ == "__main__":
    # User defined directory to write our mlruns data
    user_mlruns_dir = MLRUNS_DIR

    # User defined path to our prepared data
    train_val_data_path = os.path.join(PROCESSED_DATA_DIR, "train_val_data.csv")

    test_data_path = os.path.join(PROCESSED_DATA_DIR, "test_data.csv")

    initialize_stores(
        mlruns_dir=user_mlruns_dir, train_val_data_path=train_val_data_path, test_data_path=test_data_path
    )
    collect_config_store()
    hz.store.add_to_hydra_store(overwrite_ok=True)

    # train_single_run()
    # train_random_forest_grid_search()
    train_gradient_boosting_regressor_grid_search()

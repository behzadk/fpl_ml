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
from fpl_ml.stores import StoreGroups, HydraGroups, initialize_stores
from fpl_ml.preprocessing import DataframePipeline
from fpl_ml.train import train_sklearn
from fpl_ml.user import User
from fpl_ml.utils import set_random_seeds, delete_runs_by_metric
from projects import fpl_ml

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


def train_gradient_boosting_regressor_tpe():


    config = hz.store.get_entry("fpl_ml", "default_config")
    task_function = zen(run_train)

    # Use the project default preprocessing pipeline
    overrides = {StoreGroups.PREPROCESSING.value: fpl_ml.preprocessing.PreprocessingStores.default}

    # Set hydra overrides for tpe sampler
    overrides.update(
        {
            HydraGroups.HYDRA_SWEEPER.value: "optuna",
            HydraGroups.HYDRA_SWEEPER_SAMPLER.value: "tpe"
        }
    )

    # Set prior space for tpe sampler
    overrides.update(
        {
            StoreGroups.MODEL.value: "GradientBoostingRegressor",
            f"{StoreGroups.MODEL.value}.loss": "squared_error",
            f"{StoreGroups.MODEL.value}.learning_rate": hz.multirun([0.1,0.01,0.001]),
            f"{StoreGroups.MODEL.value}.n_estimators": hz.multirun([400,500,600,700,800]),
            f"{StoreGroups.MODEL.value}.subsample": hz.multirun([0.1,0.25,0,4,0.5,0.6]),
        }
    )

    # Run tpe sampler optimization
    (jobs,) = launch(config["node"], task_function, overrides=overrides, multirun=True)


def train_single_run():
    """Example entry script, performing a single run using the default config parameters"""
    config = hz.store.get_entry(group="fpl_ml", name="default_config")
    task_function = zen(run_train)

    # Run gridsearch
    launch(config["node"], task_function, multirun=False)


def train(user_mlruns_dir, train_val_data_path, test_data_path):

    # Initialize default stores
    initialize_stores(
        mlruns_dir=user_mlruns_dir, train_val_data_path=train_val_data_path, test_data_path=test_data_path
    )

    # Initialize default config
    collect_config_store()

    # Initialzie project specific configs
    fpl_ml.stores.initialize_project_configs()

    # Add configs to hydra store
    hz.store.add_to_hydra_store(overwrite_ok=False)

    # Run hyperparameter optimization using tpe sampler
    train_gradient_boosting_regressor_tpe()

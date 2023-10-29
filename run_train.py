
from hydra_zen import zen
from sklearn.base import RegressorMixin
from fpl_ml.config import collect_config_store, initialize_stores
import os

from fpl_ml.train import train_sklearn
from hydra_zen import launch
import mlflow
from fpl_ml.utils import set_random_seeds
import torch
from functools import partial
from typing import Union
from fpl_ml.user import User
from fpl_ml.preprocessing import DataframePipeline
import pytorch_lightning as pl


os.environ[
    "HYDRA_FULL_ERROR"
] = "1"  


def run_train(user: User, dataset: partial[torch.utils.data.Dataset], datamodule: partial[pl.LightningDataModule], 
              model: Union[RegressorMixin, torch.nn.Module], preprocessing: DataframePipeline, 
              metrics: dict, visualisations: dict):
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

    train_sklearn(datamodule, model, 
                  metrics=metrics, 
                  experiment_name="Test",
                  visualisations=visualisations,
                  return_validation_metrics=None,
                  random_seed=user.random_seed)

    # Pseudorandom so mlflow picks a new name for the next run
    set_random_seeds(None)


def train_random_forest_grid_search():
    """Example entry function, performing a grid search across several parameters of the sklearn RandomForestRegressor
    """

    config = store.get_entry("fpl_ml", "default_config")    
    task_function = zen(run_train)

    # Set parameter values we want for grid search
    overrides = ["model=RandomForestRegressor",
                 "model.n_estimators=100,200,300",
                 "model.max_features=0.1,0.25,0.5,0.75,1.0",
                 "model.max_depth=2,4,8,16",
                 "model.min_samples_split=2"]

    # Run gridsearch
    (jobs,) = launch(
        config['node'],
        task_function,
        overrides=overrides,
        multirun=True
    )

def train_gradient_boosting_regressor_grid_search():
    """Example entry function, performing a grid search across several parameters of the sklearn GradientBoostingRegressor
    """
    config = store.get_entry("fpl_ml", "default_config")    
    task_function = zen(run_train)

    # Set parameter values we want for grid search
    overrides = ["model=GradientBoostingRegressor",
                 "model.loss=squared_error",
                 "model.learning_rate=0.1,0.5",
                 "model.n_estimators=300,400,500",
                 "model.subsample=0.25,0.5"]

    # Run gridsearch
    (jobs,) = launch(
        config['node'],
        task_function,
        overrides=overrides,
        multirun=True
    )

def train_single_run():
    """Example entry script, performing a single run using the default config parameters
    """
    config = store.get_entry(group="fpl_ml", name="default_config")    
    task_function = zen(run_train)

    # Run gridsearch
    launch(
        config['node'],
        task_function,
        multirun=False
    )


if __name__ == "__main__":

    # User defined directory to write our mlruns data
    user_mlruns_dir = "/data/fpl_ml/mlruns"

    # User defined path to our prepared data
    train_val_data_path = "/data/fpl_ml/processed/all_data.csv"

    initialize_stores(mlruns_dir=user_mlruns_dir, train_val_data_path=train_val_data_path)
    store = collect_config_store()
    store.add_to_hydra_store(overwrite_ok=True)


    train_single_run()
    train_random_forest_grid_search()
    train_gradient_boosting_regressor_grid_search()

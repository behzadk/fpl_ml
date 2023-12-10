import os
import numpy as np

import hydra_zen as hz
from hydra_zen import launch, zen

from fpl_ml_projects.default_config import make_default_torch_config_store

from ml_core.stores import StoreGroups, HydraGroups, initialize_stores
from fpl_ml_projects import fpl_ml

from ml_core.train import run_train

os.environ["HYDRA_FULL_ERROR"] = "1"


def train_gradient_boosting_regressor_tpe():
    config = hz.store.get_entry("default", "default_sklearn_config")
    task_function = zen(run_train)

    # Use the project default preprocessing pipeline
    overrides = {
        StoreGroups.PREPROCESSING.value: fpl_ml.preprocessing.PreprocessingStores.default
    }

    # Set hydra overrides for tpe sampler
    overrides.update(
        {
            HydraGroups.HYDRA_SWEEPER.value: "optuna",
            HydraGroups.HYDRA_SWEEPER_SAMPLER.value: "tpe",
        }
    )

    # Set prior space for tpe sampler
    overrides.update(
        {
            StoreGroups.MODEL.value: "GradientBoostingRegressor",
            f"{StoreGroups.MODEL.value}.loss": "squared_error",
            f"{StoreGroups.MODEL.value}.learning_rate": hz.multirun([0.1, 0.01, 0.001]),
            f"{StoreGroups.MODEL.value}.n_estimators": hz.multirun(
                [400, 500, 600, 700, 800]
            ),
            f"{StoreGroups.MODEL.value}.subsample": hz.multirun(
                [0.1, 0.25, 0, 4, 0.5, 0.6]
            ),
        }
    )

    # Run tpe sampler optimization
    (jobs,) = launch(config["node"], task_function, overrides=overrides, multirun=True)


def train_torch_tpe():
    """Example entry script, performing a single run using the default config parameters"""
    config = hz.store.get_entry(group="default", name="default_torch_config")
    task_function = zen(run_train)

    # Use the project default preprocessing pipeline
    overrides = {
        f"{StoreGroups.MODEL.value}.input_dim": 59,
        f"{StoreGroups.MODEL.value}.output_dim": 1,
        StoreGroups.PREPROCESSING.value: fpl_ml.preprocessing.PreprocessingStores.default,
        StoreGroups.USER.value: "default-cpu",
    }

    # Set hydra overrides for tpe sampler
    overrides.update(
        {
            HydraGroups.HYDRA_SWEEPER.value: "optuna",
            HydraGroups.HYDRA_SWEEPER_SAMPLER.value: "tpe",
            f"+{HydraGroups.HYDRA_SWEEPER.value}/n_trials": 1,
        }
    )

    # Set prior space for tpe sampler
    overrides.update(
        {
            f"{StoreGroups.MODEL.value}.hidden_layer_sizes.0": hz.multirun(
                [5, 10, 100, 250]
            ),
            f"{StoreGroups.MODEL.value}.dropout": hz.multirun(
                [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
            ),
            f"{StoreGroups.OPTIMIZER.value}.lr": hz.multirun([0.1, 0.01, 0.001]),
            f"{StoreGroups.TRAINER.value}.max_epochs": hz.multirun(
                [1]
            ),
            f"{StoreGroups.DATAMODULE.value}.batch_size": hz.multirun(
                [512]
            ),
        }
    )

    # Run gridsearch
    launch(config["node"], task_function, overrides=overrides, multirun=True)


def train_single_torch_run():
    """Example entry script, performing a single run using the default config parameters"""
    config = hz.store.get_entry(group="default", name="default_torch_config")
    task_function = zen(run_train)

    # # Use the project default preprocessing pipeline
    # overrides = {
    #     StoreGroups.PREPROCESSING.value: fpl_ml.preprocessing.PreprocessingStores.default,
    #     StoreGroups.USER.value: "default-gpu"
    # }

    # Use the project default preprocessing pipeline
    overrides = {
        StoreGroups.PREPROCESSING.value: fpl_ml.preprocessing.PreprocessingStores.diabetes,
        StoreGroups.USER.value: "default-gpu",
    }

    # Run gridsearch
    launch(config["node"], task_function, overrides=overrides, multirun=False)


def train_demo_tpe_torch_run():
    """Example entry script, performing a single run using the default config parameters"""
    config = hz.store.get_entry(group="default", name="default_torch_config")
    task_function = zen(run_train)

    # # Use the project default preprocessing pipeline
    # overrides = {
    #     StoreGroups.PREPROCESSING.value: fpl_ml.preprocessing.PreprocessingStores.default,
    #     StoreGroups.USER.value: "default-gpu"
    # }

    # Use the project default preprocessing pipeline
    overrides = {
        f"{StoreGroups.MODEL.value}.input_dim": 10,
        f"{StoreGroups.MODEL.value}.output_dim": 1,
        StoreGroups.PREPROCESSING.value: fpl_ml.preprocessing.PreprocessingStores.diabetes,
        StoreGroups.USER.value: "default-cpu",
    }

    # Set hydra overrides for tpe sampler
    overrides.update(
        {
            HydraGroups.HYDRA_SWEEPER.value: "optuna",
            HydraGroups.HYDRA_SWEEPER_SAMPLER.value: "tpe",
            f"+{HydraGroups.HYDRA_SWEEPER.value}/n_trials": 100,
        }
    )

    # Set prior space for tpe sampler
    overrides.update(
        {
            f"{StoreGroups.MODEL.value}.hidden_layer_sizes.0": hz.multirun(
                [5, 10, 100, 250]
            ),
            f"{StoreGroups.MODEL.value}.dropout": hz.multirun(
                [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
            ),
            f"{StoreGroups.OPTIMIZER.value}.lr": hz.multirun([0.1, 0.01, 0.001]),
            f"{StoreGroups.TRAINER.value}.max_epochs": hz.multirun(
                list(np.arange(1, 100, 5))
            ),
            f"{StoreGroups.DATAMODULE.value}.batch_size": hz.multirun(
                [2, 6, 8, 12, 32, 64, 128, 256, 512]
            ),
        }
    )

    # Run gridsearch
    launch(config["node"], task_function, overrides=overrides, multirun=True)


def train(user_mlruns_dir, train_val_data_path, test_data_path):
    # Initialize default stores
    initialize_stores(
        mlruns_dir=user_mlruns_dir,
        train_val_data_path=train_val_data_path,
        test_data_path=test_data_path,
    )

    # Initialize default config
    make_default_torch_config_store()

    # Initialzie project specific configs
    fpl_ml.stores.initialize_project_configs()

    # Add configs to hydra store
    hz.store.add_to_hydra_store(overwrite_ok=False)

    # Run hyperparameter optimization using tpe sampler
    # train_single_torch_run()
    train_torch_tpe()
    # train_gradient_boosting_regressor_tpe()


def train_demo(user_mlruns_dir, train_val_data_path, test_data_path):
    # Initialize default stores
    initialize_stores(
        mlruns_dir=user_mlruns_dir,
        train_val_data_path=train_val_data_path,
        test_data_path=test_data_path,
    )

    # Initialize default config
    make_default_torch_config_store()

    # Initialzie project specific configs
    fpl_ml.stores.initialize_project_configs()

    # Add configs to hydra store
    hz.store.add_to_hydra_store(overwrite_ok=False)

    # Run hyperparameter optimization using tpe sampler
    # train_single_torch_run()
    train_demo_tpe_torch_run()
    # train_torch_tpe()
    # train_gradient_boosting_regressor_tpe()

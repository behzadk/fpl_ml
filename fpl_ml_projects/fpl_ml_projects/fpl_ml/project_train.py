import os

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


def train_torch_tpe(user_mlruns_dir, train_val_data_path, test_data_path):
    """Example entry script, performing a single run using the default config parameters"""

    # Initialize default stores and default config
    initialize_stores(
        mlruns_dir=user_mlruns_dir,
        train_val_data_path=train_val_data_path,
        test_data_path=test_data_path,
    )
    make_default_torch_config_store()

    # Initialzie project specific configs
    fpl_ml.stores.initialize_project_configs()
    hz.store.add_to_hydra_store(overwrite_ok=False)  #

    # Use the project default preprocessing pipeline
    overrides = {
        f"{StoreGroups.MODEL.value}.input_dim": 79,
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
            f"{StoreGroups.MODEL.value}.hidden_layer_sizes": hz.multirun(
                [[500], [100], [50, 25], [100, 50, 25], [100, 50], [250, 50]]
            ),
            f"{StoreGroups.MODEL.value}.dropout": hz.multirun([0.2, 0.3, 0.5]),
            f"{StoreGroups.OPTIMIZER.value}.lr": hz.multirun([0.01, 0.001, 0.0001]),
            f"{StoreGroups.TRAINER.value}.max_epochs": hz.multirun([50]),
            f"{StoreGroups.DATAMODULE.value}.batch_size": hz.multirun(
                [16, 32, 64, 128, 512]
            ),
        }
    )

    config = hz.store.get_entry(group="default", name="default_torch_config")
    task_function = zen(run_train)

    # Run gridsearch
    launch(config["node"], task_function, overrides=overrides, multirun=True)


def train_single_torch_run():
    """Example entry script, performing a single run using the default config parameters"""
    config = hz.store.get_entry(group="default", name="default_torch_config")
    task_function = zen(run_train)

    # Use the project default preprocessing pipeline
    overrides = {
        StoreGroups.PREPROCESSING.value: fpl_ml.preprocessing.PreprocessingStores.diabetes,
        StoreGroups.USER.value: "default-gpu",
    }

    # Run gridsearch
    launch(config["node"], task_function, overrides=overrides, multirun=False)

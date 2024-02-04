import os

import hydra_zen as hz
from hydra_zen import launch, zen

from fpl_ml_projects.default_config import (
    make_default_torch_config_store,
    make_default_sklearn_config_store,
)

from ml_core.stores import StoreGroups, HydraGroups, initialize_stores
from fpl_ml_projects import fpl_ml

from ml_core.train import run_train
import numpy as np

os.environ["HYDRA_FULL_ERROR"] = "1"



def _build_random_forest_overrides():
    overrides = {
        StoreGroups.MODEL.value: "RandomForestRegressor"
    }
    # Override hyperparameters
    overrides.update(
        {
            f"{StoreGroups.MODEL.value}.n_estimators": hz.multirun(
                np.arange(50, 300, step=50)
            ),
            f"{StoreGroups.MODEL.value}.min_samples_split": hz.multirun(
                np.arange(0.1, 1.1, step=0.1)
            ),
            f"{StoreGroups.MODEL.value}.max_features": hz.multirun(
                np.arange(0.1, 1.1, step=0.1)
            ),
            f"{StoreGroups.MODEL.value}.max_samples": hz.multirun(
                np.arange(0.1, 1.1, step=0.1)
            ),
            f"{StoreGroups.MODEL.value}.criterion": hz.multirun(["absolute_error", "friedman_mse"])
        }
    )

    return overrides


def _build_svm_overrides():
    overrides = {
        StoreGroups.MODEL.value: "LinearSVR"
    }

    # Override hyperparameters
    overrides.update(
        {
            f"{StoreGroups.MODEL.value}.epsilon": hz.multirun([0.0, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]),
            f"{StoreGroups.MODEL.value}.tol": hz.multirun(
                [1e-2, 1e-1, 1e-4, 1e-3, 1e-5]
            ),
            f"{StoreGroups.MODEL.value}.C": hz.multirun(
                [0.01, 0.25, 1.0, 0.5, 0.1, 1.5]
            )
        }
    )

    return overrides


def train_svm(user_mlruns_dir, train_val_data_path, test_data_path):
    # Initialize default stores and default config
    initialize_stores(
        mlruns_dir=user_mlruns_dir,
        train_val_data_path=train_val_data_path,
        test_data_path=test_data_path,
    )
    make_default_sklearn_config_store()

    # Initialzie project specific configs
    fpl_ml.stores.initialize_project_configs()
    hz.store.add_to_hydra_store(overwrite_ok=False)

    config = hz.store.get_entry("default", "default_sklearn_config")
    task_function = zen(run_train)

    overrides = _build_svm_overrides()

    # Use the project default preprocessing pipeline
    overrides.update(
        {   "experiment_name": "random_forest_regressor_svm",
            StoreGroups.PREPROCESSING.value: fpl_ml.preprocessing.PreprocessingStores.default
        }
    )


    # Set hydra overrides for tpe sampler
    overrides.update(
        {
            HydraGroups.HYDRA_SWEEPER.value: "optuna",
            HydraGroups.HYDRA_SWEEPER_SAMPLER.value: "tpe",
            "hydra.sweeper.n_trials": 100,
        }
    )


    # Run tpe sampler optimization
    launch(config["node"], task_function, overrides=overrides, multirun=True)



def train_gradient_boosting_regressor_tpe(
    user_mlruns_dir, train_val_data_path, test_data_path
):
    # Initialize default stores and default config
    initialize_stores(
        mlruns_dir=user_mlruns_dir,
        train_val_data_path=train_val_data_path,
        test_data_path=test_data_path,
    )
    make_default_sklearn_config_store()

    # Initialzie project specific configs
    fpl_ml.stores.initialize_project_configs()
    hz.store.add_to_hydra_store(overwrite_ok=False)

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
            f"{StoreGroups.MODEL.value}.loss": "absolute_error",
            f"{StoreGroups.MODEL.value}.learning_rate": hz.multirun([0.1, 0.01, 0.001]),
            f"{StoreGroups.MODEL.value}.n_estimators": hz.multirun(
                [400, 500, 600, 700, 800]
            ),
            f"{StoreGroups.MODEL.value}.subsample": hz.multirun(
                [0.1, 0.25, 0.4, 0.5, 0.6]
            ),
        }
    )

    # Run tpe sampler optimization
    launch(config["node"], task_function, overrides=overrides, multirun=True)


def train_random_forest_regressor(user_mlruns_dir, train_val_data_path, test_data_path):
    # Initialize default stores and default config
    initialize_stores(
        mlruns_dir=user_mlruns_dir,
        train_val_data_path=train_val_data_path,
        test_data_path=test_data_path,
    )
    make_default_sklearn_config_store()

    # Initialzie project specific configs
    fpl_ml.stores.initialize_project_configs()
    hz.store.add_to_hydra_store(overwrite_ok=False)

    config = hz.store.get_entry("default", "default_sklearn_config")
    task_function = zen(run_train)

    overrides = _build_random_forest_overrides()

    # Use the project default preprocessing pipeline
    overrides.update(
        {   "experiment_name": "random_forest_regressor_2",
            StoreGroups.PREPROCESSING.value: fpl_ml.preprocessing.PreprocessingStores.default
        }
    )


    # Set hydra overrides for tpe sampler
    overrides.update(
        {
            HydraGroups.HYDRA_SWEEPER.value: "optuna",
            HydraGroups.HYDRA_SWEEPER_SAMPLER.value: "tpe",
        }
    )


    # Run tpe sampler optimization
    launch(config["node"], task_function, overrides=overrides, multirun=True)



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
    hz.store.add_to_hydra_store(overwrite_ok=False)

    # Use the project default preprocessing pipeline
    overrides = {
        f"{StoreGroups.MODEL.value}.input_dim": 71,
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
            "hydra.sweeper.direction": hz.hydra_list(["maximize"]),
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

    overrides.update({StoreGroups.LOSS.value: "MAE"})

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

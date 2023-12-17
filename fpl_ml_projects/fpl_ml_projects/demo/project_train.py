import os
import numpy as np

import hydra_zen as hz
from hydra_zen import launch, zen

from fpl_ml_projects.default_config import make_default_torch_config_store

from ml_core.stores import StoreGroups, HydraGroups, initialize_stores
from fpl_ml_projects import fpl_ml

from ml_core.train import run_train

os.environ["HYDRA_FULL_ERROR"] = "1"


def train_demo_tpe_torch_run(user_mlruns_dir, train_val_data_path, test_data_path):
    """Example entry script, performing a single run using the default config parameters"""

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

    config = hz.store.get_entry(group="default", name="default_torch_config")

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
    launch(config["node"], zen(run_train), overrides=overrides, multirun=True)

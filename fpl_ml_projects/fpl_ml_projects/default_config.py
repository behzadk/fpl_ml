from hydra_zen import ZenField
import hydra_zen as hz
from dataclasses import dataclass
from typing import Any

from ml_core.stores import StoreGroups


@dataclass
class SklearnBaseConfig:
    # Must be passed at command line -- neccesary arguments
    user: Any = hz.MISSING
    dataset: Any = hz.MISSING
    datamodule: Any = hz.MISSING
    model: Any = hz.MISSING
    preprocessing: Any = hz.MISSING
    data_splitter: Any = hz.MISSING
    metrics: Any = hz.MISSING
    visualisations: Any = hz.MISSING


@dataclass
class TorchBaseConfig:
    # Must be passed at command line -- neccesary arguments
    user: Any = hz.MISSING
    dataset: Any = hz.MISSING
    datamodule: Any = hz.MISSING
    preprocessing: Any = hz.MISSING
    data_splitter: Any = hz.MISSING
    metrics: Any = hz.MISSING
    visualisations: Any = hz.MISSING

    model: Any = hz.MISSING
    optimizer: Any = hz.MISSING
    trainer: Any = hz.MISSING
    callbacks: Any = hz.MISSING
    loss: Any = hz.MISSING


def make_default_sklearn_config_store():
    zen_config = []

    for value in SklearnBaseConfig.__dataclass_fields__.values():
        item = (
            ZenField(name=value.name, hint=value.type, default=value.default)
            if value.default is not hz.MISSING
            else ZenField(name=value.name, hint=value.type)
        )
        zen_config.append(item)

    config = hz.make_config(
        *zen_config,
        hydra_defaults=[
            "_self_",
            {StoreGroups.USER.value: "default-cpu"},
            {StoreGroups.DATASET.value: "default-sklearn"},
            {StoreGroups.DATAMODULE.value: "default"},
            {StoreGroups.MODEL.value: "default_sklearn"},
            {StoreGroups.PREPROCESSING.value: "default"},
            {StoreGroups.DATA_SPLITTER.value: "default"},
            {StoreGroups.METRICS.value: "regression_default"},
            {StoreGroups.VISUALISATIONS.value: "regression_default"},
        ]
    )

    hz.store(config, group="default", name="default_sklearn_config")
    hz.store.add_to_hydra_store(overwrite_ok=False)


def make_default_torch_config_store():
    zen_config = []

    for value in TorchBaseConfig.__dataclass_fields__.values():
        item = (
            ZenField(name=value.name, hint=value.type, default=value.default)
            if value.default is not hz.MISSING
            else ZenField(name=value.name, hint=value.type)
        )
        zen_config.append(item)

    config = hz.make_config(
        *zen_config,
        hydra_defaults=[
            "_self_",
            {StoreGroups.USER.value: "default-cpu"},
            {StoreGroups.DATASET.value: "default"},
            {StoreGroups.DATAMODULE.value: "default-torch"},
            {StoreGroups.PREPROCESSING.value: "default"},
            {StoreGroups.DATA_SPLITTER.value: "default"},
            {StoreGroups.METRICS.value: "regression_default"},
            {StoreGroups.VISUALISATIONS.value: "regression_default"},
            {StoreGroups.MODEL.value: "neural_network"},
            {StoreGroups.OPTIMIZER.value: "default"},
            {StoreGroups.TRAINER.value: "default"},
            {StoreGroups.CALLBACKS.value: "default"},
            {StoreGroups.LOSS.value: "MSE"},
        ]
    )

    hz.store(config, group="default", name="default_torch_config")
    hz.store.add_to_hydra_store(overwrite_ok=False)

from hydra_zen import ZenField
import hydra_zen as hz
from dataclasses import dataclass
from typing import Any


@dataclass
class BaseConfig:
    # Must be passed at command line -- neccesary arguments
    user: Any = hz.MISSING
    dataset: Any = hz.MISSING
    datamodule: Any = hz.MISSING
    model: Any = hz.MISSING
    preprocessing: Any = hz.MISSING
    data_splitter: Any = hz.MISSING
    metrics: Any = hz.MISSING
    visualisations: Any = hz.MISSING


def collect_config_store():
    zen_config = []

    for value in BaseConfig.__dataclass_fields__.values():
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

    hz.store(config, group="default", name="default_config")
    hz.store.add_to_hydra_store(overwrite_ok=False)

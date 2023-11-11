import hydra_zen as hz
from ml_core.stores import StoreGroups
from ml_core.preprocessing import (
    StandardScaleColumns,
    OneHotEncodeColumns,
    SplitFeaturesAndLabels,
    DataframePipeline,
)
from projects.fpl_ml import constants


class PreprocessingStores:
    default = "fpl_ml"


def initialize_preprocessing_config():
    preprocessing_store = hz.store(group=StoreGroups.PREPROCESSING.value)

    numerical_features = constants.NumericalFeatures.ALL
    categorical_features = constants.CategoricalFeatures.ALL

    steps = []

    # Scale numerical features
    scale_step = hz.builds(
        StandardScaleColumns,
        scale_columns=numerical_features,
        scale_column_prefixes=[constants.Prefixes.ROLLING_AVERAGE],
        hydra_convert="all",
    )

    # Onehot encode categorical features
    encode_step = hz.builds(
        OneHotEncodeColumns, columns_to_encode=categorical_features, hydra_convert="all"
    )

    # Split features and labels using the X prefix and total_points label
    split_step = hz.builds(
        SplitFeaturesAndLabels,
        x_column_prefixes=[constants.Prefixes.X],
        y_columns=[constants.Labels.TOTAL_POINTS],
        hydra_convert="all",
    )

    steps = [scale_step, encode_step, split_step]

    preprocessing_store(
        DataframePipeline,
        dataframe_processing_steps=steps,
        name=PreprocessingStores.default,
    )

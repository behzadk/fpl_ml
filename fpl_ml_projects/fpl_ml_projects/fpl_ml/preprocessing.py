import hydra_zen as hz
from ml_core.stores import StoreGroups
from ml_core.preprocessing import (
    StandardScaleColumns,
    OneHotEncodeColumns,
    SplitFeaturesAndLabels,
    DataframePipeline,
)
from fpl_ml_projects.fpl_ml import constants


class PreprocessingStores:
    default = "fpl_ml"
    scaled = "fpl_ml_scaled"
    diabetes = "diabetes"


def initialize_preprocessing_config():
    preprocessing_store = hz.store(group=StoreGroups.PREPROCESSING.value)

    numerical_features = constants.NumericalFeatures.ALL
    categorical_features = constants.CategoricalFeatures.ALL

    # Scale numerical features
    scale_step = hz.builds(
        StandardScaleColumns,
        scale_columns=numerical_features,
        scale_column_prefixes=[
            constants.Prefixes.ROLLING_AVERAGE,
            constants.Prefixes.GAME_WEEK_AVERAGE,
        ],
        hydra_convert="all",
    )

    # Onehot encode categorical features
    encode_step = hz.builds(
        OneHotEncodeColumns, columns_to_encode=categorical_features, hydra_convert="all"
    )

    # Split features and labels using the X prefix and total_points label
    split_step = hz.builds(
        SplitFeaturesAndLabels,
        x_column_prefixes=[
            f"{constants.Prefixes.X}value",
            f"{constants.Prefixes.X}opponent_team",
            f"{constants.Prefixes.X}element_type",
            f"{constants.Prefixes.X}rolling_",
        ],
        y_columns=[constants.Labels.TOTAL_POINTS],
        hydra_convert="all",
    )

    preprocessing_store(
        DataframePipeline,
        dataframe_processing_steps=[encode_step, split_step],
        name=PreprocessingStores.default,
    )
    preprocessing_store(
        DataframePipeline,
        dataframe_processing_steps=[scale_step, encode_step, split_step],
        name=PreprocessingStores.scaled,
    )

    # Split features and labels using the X prefix and total_points label
    diabetes_split_step = hz.builds(
        SplitFeaturesAndLabels,
        x_column_prefixes=[constants.Prefixes.X],
        y_columns=["Y_target"],
        hydra_convert="all",
    )

    preprocessing_store(
        DataframePipeline,
        dataframe_processing_steps=[diabetes_split_step],
        name=PreprocessingStores.diabetes,
    )

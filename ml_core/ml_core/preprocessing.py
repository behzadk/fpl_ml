from abc import ABCMeta, abstractmethod

import pandas as pd
from loguru import logger as _LOGGER
from mlflow.pyfunc import PythonModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_core.utils import get_columns_with_prefix


class DataSplitter(metaclass=ABCMeta):
    @abstractmethod
    def split(self, df: pd.DataFrame):
        return


class DataframeProcessingStep(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, df):
        pass

    @abstractmethod
    def transform(self, df) -> pd.DataFrame:
        pass

    @abstractmethod
    def fit_transform(self, df) -> pd.DataFrame:
        pass


class DataframePipeline(PythonModel):
    def __init__(self, dataframe_processing_steps: list[DataframeProcessingStep]):
        self.steps = dataframe_processing_steps

    def fit(self, df: pd.DataFrame):
        for step in self.steps:
            step.fit(df)

    def transform(self, df: pd.DataFrame):
        for step in self.steps:
            _LOGGER.info(f"Transform: {step}")
            df = step.transform(df)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for step in self.steps:
            _LOGGER.info(f"Pipeline fit transform: {step}")
            step.fit(df)
            df = step.transform(df)

        return df


class Identity(DataframeProcessingStep):
    def __init__(self):
        pass

    def fit(self):
        pass

    def transform(self):
        pass


class SplitFeaturesAndLabels(DataframeProcessingStep):
    def __init__(self, x_column_prefixes: list[str], y_columns: list[str]):
        self.x_column_prefixes = x_column_prefixes
        self.y_columns = y_columns

    def fit(self, df):
        pass

    def transform(self, df: pd.DataFrame) -> dict:
        x_columns = get_columns_with_prefix(df, prefix_list=self.x_column_prefixes)

        X = df[x_columns].values
        y = df[self.y_columns].values

        return {"X": X, "y": y}

    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        df = self.transform(df)

        return df


class StandardScaleColumns(DataframeProcessingStep):
    def __init__(self, scale_columns, scale_column_prefixes):
        self._scale_columns = scale_columns
        self._scale_column_prefixes = scale_column_prefixes
        self._scaler = StandardScaler()

    def fit(self, df):
        if self._scale_columns:
            scale_columns = self._scale_columns[:]

        else:
            scale_columns = []

        if self._scale_column_prefixes:
            scale_columns.extend(
                get_columns_with_prefix(df, self._scale_column_prefixes)
            )

        self._scale_columns = list(set(scale_columns))

        X = df[self._scale_columns].values

        self._scaler.fit(X)

    def transform(self, df) -> pd.DataFrame:
        X = df[self._scale_columns].values
        self._scaler.transform(X)

        for idx, x in enumerate(self._scale_columns):
            df[x] = X[:, idx]

        return df

    def fit_transform(self, df) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)


class OneHotEncodeColumns(DataframeProcessingStep):
    def __init__(self, columns_to_encode: list[str], drop_encoded_column: bool = True):
        self.columns_to_encode = columns_to_encode
        self.drop_encoded_column = drop_encoded_column

    def fit(self, df: pd.DataFrame):
        encoders = []
        for col_name in self.columns_to_encode:
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
            encoder.fit(df[[col_name]])
            encoders.append(encoder)

        self.encoders = encoders

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for idx, col_name in enumerate(self.columns_to_encode):
            encoded_df = pd.DataFrame(self.encoders[idx].transform(df[[col_name]]))
            encoded_df.columns = self.encoders[idx].get_feature_names_out([col_name])
            encoded_df.index = df.index

            df = pd.merge(df, encoded_df, left_index=True, right_index=True)

            if self.drop_encoded_column:
                df.drop(col_name, inplace=True, axis=1)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)


class RandomSplitData(DataSplitter):
    def __init__(self, frac):
        self.frac = frac

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.frac == 1.0:
            return df, df[0:0]

        elif self.frac == 0.0:
            return df[0:0], df

        df_a, df_b = train_test_split(df, test_size=1 - self.frac, shuffle=True)

        return df_a, df_b

from typing import Callable

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from fpl_ml.preprocessing import DataframePipeline


class DataModuleLoadedFromCSV(pl.LightningDataModule):
    """Datamodule used to handle the data staging, splitting and preprocessing"""

    def __init__(
        self,
        train_validation_splitter: Callable,
        preprocessing_pipeline: DataframePipeline,
        partial_dataset: type[torch.utils.data.Dataset],
        train_validation_data_path: str | None = None,
        test_data_path: str | None = None,
        predict_data_path: str | None = None,
        batch_size: int | None = 32,
        use_torch: bool = False,
        device: str = "cpu",
    ):
        super().__init__()

        #  Set data paths
        self.train_validation_data_path = train_validation_data_path
        self.test_data_path = test_data_path

        self._partial_dataset = partial_dataset

        self.predict_data_path = predict_data_path

        self.train_validation_splitter = train_validation_splitter
        self.preprocessing_pipeline = preprocessing_pipeline

        self._batch_size = batch_size
        self._use_torch = use_torch
        self._device = device

    def download_data(self):
        pass

    def setup(self, stage: str):
        if stage == "fit":
            assert (
                self.train_validation_data_path is not None
            ), "Train/validation data path not set and is required for fitting stage"

            train_val_df = pd.read_csv(self.train_validation_data_path)
            train_df, val_df = self.train_validation_splitter.split(train_val_df)

            # Make training and validation datasets
            self.train_dataset = self._partial_dataset(
                **self.preprocessing_pipeline.fit_transform(train_df),
                use_torch=self._use_torch,
                device=self._device,
            )
            self.val_dataset = self._partial_dataset(
                **self.preprocessing_pipeline.transform(val_df),
                use_torch=self._use_torch,
                device=self._device,
            )

        elif stage == "test":
            assert (
                self.test_data_path is not None
            ), "Test data path not set and is required for staging test"

            test_df = pd.read_csv(self.test_data_path)

            self.test_dataset = self._partial_dataset(
                **self.preprocessing_pipeline.transform(test_df),
                use_torch=self._use_torch,
                device=self._device,
            )

        elif stage == "predict":
            assert (
                self.test_data_path is not None
            ), "Predict data path not set and is required for staging predict"

            predict_df = pd.read_csv(self.predict_data_path)

            pred_dict = self.preprocessing_pipeline.transform(predict_df)

            self.predict_dataset = self._partial_dataset(
                **pred_dict,
                predict=True,
                use_torch=self._use_torch,
                device=self._device,
            )

        else:
            raise ValueError(f"stage {stage} not recognised")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self._batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self._batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self._batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self._batch_size)

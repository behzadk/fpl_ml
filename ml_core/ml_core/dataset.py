import numpy as np
import torch
from loguru import logger


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        features_key: str = "X",
        labels_key: str = "y",
        predict: bool = False,
        use_torch: bool = False,
        device: str = "cpu",
        precision: int = 32
    ):
        self._X = X
        self._y = y

        self._features_key = features_key
        self._labels_key = labels_key

        if use_torch:
            # Move all data to device
            self._X = torch.from_numpy(self._X.astype(f'float{precision}')).to(device)
            self._y = torch.from_numpy(self._y.astype(f'float{precision}')).to(device)

            logger.info(self._X.dtype)

        self.predict = predict

    def get_input_dim(self):
        return self._X[0].shape

    def get_output_dim(self):
        return self._y[0].shape

    def __len__(self):
        return self._X.shape[0]

    def __getitem__(self, idx):
        output = {}
        if self.predict:
            output[self._features_key] = self._X[idx]
            return output

        output[self._features_key] = self._X[idx]
        output[self._labels_key] = self._y[idx]

        return output
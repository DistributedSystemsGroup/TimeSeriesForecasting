from abc import ABC, abstractmethod
from typing import List

import numpy as np
import more_itertools

from core.AbstractForecastingModel import AbstractForecastingModel
from core.Prediction import Prediction


class AbstractRecursiveForecastingModel(AbstractForecastingModel, ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def predict(self, future_points: int = 1) -> List[Prediction]:
        raise NotImplementedError

    def build_train_test(self, train_window: int = 1):
        train_x = np.array(list(more_itertools.windowed(self.time_series_values[0:-1], n=train_window)))
        train_y = np.reshape(self.time_series_values[train_window:], (-1, 1))
        test_x = np.reshape(self.time_series_values[len(self.time_series_values)-train_window:], (1, -1))

        return train_x, train_y, test_x

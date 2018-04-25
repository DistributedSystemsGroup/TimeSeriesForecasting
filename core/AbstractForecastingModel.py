from abc import ABC, abstractmethod
from typing import List, Type

import numpy as np

from core.Prediction import Prediction


class AbstractForecastingModel(ABC):
    def __init__(self, history_size=None):
        self.history_size = history_size
        self.time_series_values = []

    def __slice_history__(self):
        if self.history_size is not None:
            self.time_series_values = self.time_series_values[-self.history_size:]

    def add_observations(self, values: List[float]):
        self.time_series_values = np.concatenate((self.time_series_values, values))
        self.__slice_history__()

    def add_observation(self, value: float):
        self.time_series_values.append(value)
        self.__slice_history__()

    def get_observations(self) -> List[float]:
        return self.time_series_values

    def reset(self):
        self.time_series_values = []

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def predict(self, future_points: int = 1) -> List[Prediction]:
        raise NotImplementedError



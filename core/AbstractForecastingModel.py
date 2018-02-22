from abc import ABC, abstractmethod
from typing import List

import numpy as np

from core.Prediction import Prediction


class AbstractForecastingModel(ABC):
    def __init__(self):
        self.time_series_values = []

    def add_observations(self, values: List[float]):
        self.time_series_values = np.concatenate((self.time_series_values, values), axis=0)

    def add_observation(self, value: float):
        self.time_series_values.append(value)

    def get_observations(self) -> List[float]:
        return self.time_series_values

    def reset(self):
        self.time_series_values = []

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def predict(self, future_points: int = 1) ->List[Prediction]:
        pass



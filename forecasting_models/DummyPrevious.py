from typing import List

import numpy as np

from core.AbstractForecastingModel import AbstractForecastingModel
from core.Prediction import Prediction


class DummyPrevious(AbstractForecastingModel):
    @property
    def name(self):
        return "DummyPrevious"

    def predict(self, future_points: int = 1) -> List[Prediction]:
        return [Prediction(self.time_series_values[len(self.time_series_values) - 1]) for i in np.arange(future_points)]

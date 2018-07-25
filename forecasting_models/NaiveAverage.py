from typing import List

import numpy as np

from core.AbstractForecastingModel import AbstractForecastingModel
from core.Prediction import Prediction


class NaiveAverage(AbstractForecastingModel):
    @property
    def name(self):
        return "NaiveAverage"

    def predict(self, future_points: int = 1) -> List[Prediction]:
        return [Prediction(np.mean(self.time_series_values)) for i in np.arange(future_points)]

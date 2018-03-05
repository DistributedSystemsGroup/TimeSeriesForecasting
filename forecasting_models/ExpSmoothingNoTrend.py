from typing import List

import numpy as np

from core.AbstractForecastingModel import AbstractForecastingModel
from core.Prediction import Prediction

alpha = 0.9


class ExpSmoothingNoTrend(AbstractForecastingModel):
    @property
    def name(self) -> str:
        return "ExpSmoothingNoTrend"

    def predict(self, future_points: int = 1) -> List[Prediction]:
        # Error correction form
        result = [self.time_series_values[0]]
        error = 0
        for i in range(0, len(self.time_series_values) + future_points - 1):
            if i < len(self.time_series_values):
                error = self.time_series_values[i] - result[i]
            result.append(result[i] + alpha * error)

        # possibility to get the whole estimated series (comparison)
        predictions = result[len(self.time_series_values):]

        return [Prediction(predictions[i]) for i in np.arange(len(predictions))]

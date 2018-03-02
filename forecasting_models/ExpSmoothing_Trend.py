from typing import List

import numpy as np

from core.AbstractForecastingModel import AbstractForecastingModel
from core.Prediction import Prediction

alpha = 0.9
gamma = 0.9

class ExpSmoothing_Trend(AbstractForecastingModel):
    @property
    def name(self) -> str:
        return "ExpSmoothing_Trend"

    def predict(self, future_points: int = 1) -> List[Prediction]:

    # Error correction form

        # Inizialization:
        results = [self.time_series_values[0]]
        level = self.time_series_values[0]
        trend = self.time_series_values[1] - self.time_series_values[0]

        for i in range(1, len(self.time_series_values)+future_points):
            if i < len(self.time_series_values):
                error = self.time_series_values[i-1] - results[i-1]
            level = level + trend + alpha * error
            trend = trend + alpha * gamma * error
            results.append(level + trend)

        predictions = results[len(self.time_series_values):]

        return [Prediction(predictions[i]) for i in np.arange(len(predictions))]


    # def predict(self, future_points: int = 1, observations=[]) -> List[Prediction]:
    #
    # # Component (recurrence) form
    #
    #     # Inizialization:
    #     result = [self.time_series_values[0]]
    #     level = self.time_series_values[0]
    #     trend = self.time_series_values[1] - self.time_series_values[0]
    #
    #     for i in range(1, len(self.time_series_values)+future_points):
    #         if i < len(self.time_series_values):
    #             value = self.time_series_values[i]
    #         else:
    #             value = result[i-1]
    #         last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
    #         trend = gamma * (level - last_level) + (1 - gamma) * trend
    #         result.append(level + trend)
    #
    #     predictions = result[len(self.time_series_values):]
    #
    #     return [Prediction(predictions[i]) for i in np.arange(len(predictions))]


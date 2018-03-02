from typing import List

import numpy as np

from core.AbstractForecastingModel import AbstractForecastingModel
from core.Prediction import Prediction

alpha = 0.9

class ExpSmoothing_NoTrend(AbstractForecastingModel):
    @property
    def name(self) -> str:
        return "ExpSmoothing_NoTrend"

    def predict(self, future_points: int = 1) -> List[Prediction]:  #need to retrieve true observations (for now parameters of the method coming from experiment.py)

    # Error correction form

        result = [self.time_series_values[0]]
        for i in range(0, len(self.time_series_values)+future_points-1):
            if i < len(self.time_series_values):
                error = self.time_series_values[i] - result[i]
            result.append(result[i]+alpha*error)

        predictions = result[len(self.time_series_values):]  # possibility to get the whole estimated series (comparison)

        return [Prediction(predictions[i]) for i in np.arange(len(predictions))]



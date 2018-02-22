from typing import List

import numpy as np

from core.AbstractForecastingModel import AbstractForecastingModel
from core.Prediction import Prediction

from statsmodels.tsa.arima_model import ARIMA

p = 0
d = 1
q = 1


class Arima(AbstractForecastingModel):
    @property
    def name(self) -> str:
        return "Arima"

    def predict(self, future_points: int = 1) -> List[Prediction]:
        model = ARIMA(self.time_series_values, order=(p, d, q))
        model_fit = model.fit(disp=0)

        predictions, std_errors, confidence_intervals = model_fit.forecast(steps=future_points)

        return [Prediction(predictions[i], std_errors[i] * np.sqrt(len(self.time_series_values)))
                for i in np.arange(len(predictions))]

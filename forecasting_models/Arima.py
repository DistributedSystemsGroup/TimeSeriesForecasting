from typing import List

import numpy as np
from pyramid.arima import auto_arima

from core.AbstractForecastingModel import AbstractForecastingModel
from core.Prediction import Prediction

class Arima(AbstractForecastingModel):
    @property
    def name(self) -> str:
        return "Arima"


    def predict(self, future_points: int = 1) -> List[Prediction]:

        model = auto_arima(self.time_series_values, start_p=0, start_q=0, max_p=3, max_d=3, max_q=3, error_action="ignore", suppress_warnings=True)

        predictions = model.predict(n_periods=future_points)

        return [Prediction(predictions[i]) for i in np.arange(len(predictions))]

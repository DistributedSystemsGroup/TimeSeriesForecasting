from typing import List

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from core.AbstractForecastingModel import AbstractForecastingModel
from core.Prediction import Prediction

class ExpSmoothing(AbstractForecastingModel):
    @property
    def name(self) -> str:
        return "ExpSmoothing"

    def predict(self, future_points: int = 1) -> List[Prediction]:

        exp_smooth = ExponentialSmoothing(np.array(self.time_series_values), trend="add")
        model = exp_smooth.fit()
        predictions = model.forecast(future_points)

        return [Prediction(predictions[i]) for i in np.arange(len(predictions))]
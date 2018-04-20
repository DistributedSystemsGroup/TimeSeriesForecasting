from typing import List

import numpy as np
from numpy.linalg import LinAlgError
from pyramid.arima import auto_arima
from pyramid.arima import ARIMA as pArima

from core.AbstractForecastingModel import AbstractForecastingModel
from core.Prediction import Prediction
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

d = 1
seasonal = False
maxiter = 25

OPTIMIZE_EVERY = 100


class AutoArima(AbstractForecastingModel):
    def __init__(self, *args, **kwargs):
        super(__class__, self).__init__(*args, **kwargs)
        self.last_optimization = 0
        self.model = None

    @property
    def name(self) -> str:
        return "AutoArima"

    def auto_arima(self):
        m = auto_arima(self.time_series_values, start_p=0, start_q=0,  # d=d,
                       max_p=3, max_d=3, max_q=3,  # max_order=None,
                       maxiter=maxiter,  seasonal=seasonal,
                       error_action="ignore", suppress_warnings=True)
        self.last_optimization = 0
        print("Optimized ARIMA parameters: {}".format(m.get_params()["order"]))
        return m

    def predict(self, future_points: int = 1) -> List[Prediction]:
        if self.last_optimization == 0:
            self.model = self.auto_arima()
        else:
            try:
                self.model = self.model.fit(self.time_series_values)
            except (LinAlgError, ValueError):
                print("Error from the fitting, rerunning the ARIMA optimization")
                self.auto_arima()

        self.last_optimization = (self.last_optimization + 1) % OPTIMIZE_EVERY

        predictions = self.model.predict(n_periods=future_points)

        return [Prediction(predictions[i]) for i in np.arange(len(predictions))]

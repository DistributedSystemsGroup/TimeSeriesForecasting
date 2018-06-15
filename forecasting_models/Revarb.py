from typing import List

import numpy as np
from sklearn.preprocessing import StandardScaler
import GPy
from libs.RGP import autoreg

from core.AbstractForecastingModel import AbstractForecastingModel
from core.Prediction import Prediction

train_window = 2

np.random.seed(1234)

class Revarb(AbstractForecastingModel):
    @property
    def name(self) -> str:
        return "Revarb"

    def build_train_test(self, future_points: int = 1):

        self.scaler = StandardScaler()
        self.scaler.fit(np.reshape(self.time_series_values, (-1, 1)))
        scaled_series = self.scaler.transform(np.reshape(self.time_series_values, (-1, 1)))

        train_X = np.arange(len(scaled_series),dtype=float).reshape((-1,1))
        train_Y = np.reshape(scaled_series, (-1, 1))
        test_X = np.arange(len(scaled_series),len(scaled_series)+future_points+train_window).reshape((-1,1))

        return train_X, train_Y, test_X

    def predict(self, future_points: int = 1) -> List[Prediction]:

        train_X, train_Y, test_X = self.build_train_test(future_points)

        win_out = 1
        win_in = train_window
        model = autoreg.DeepAutoreg([0,win_out], train_Y, U=train_X, U_win=win_in,
                                    kernels=[GPy.kern.RBF(win_out,ARD=True,variance=0.8,lengthscale=4),
                                    GPy.kern.RBF(win_in + win_out,ARD=True,variance=0.8,lengthscale=4)])

        model.optimize(messages=1,max_iters=70)

        posterior_pred = model.freerun(U=test_X)
        result = posterior_pred.mean
        vars = posterior_pred.variance

        predictions = self.scaler.inverse_transform(result).flatten()
        variances = self.scaler.inverse_transform(vars).flatten()

        return [Prediction(predictions[i],variances[i]) for i in np.arange(len(predictions))]

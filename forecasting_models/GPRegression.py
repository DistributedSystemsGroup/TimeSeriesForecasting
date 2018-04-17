from typing import List

import numpy as np
import more_itertools
from sklearn.preprocessing import StandardScaler
import gpflow

from core.AbstractForecastingModel import AbstractForecastingModel
from core.Prediction import Prediction

train_window = 2

np.random.seed(1234)

class GPRegression(AbstractForecastingModel):
    @property
    def name(self) -> str:
        return "GPRegression"

    def build_train_test(self):

        self.scaler = StandardScaler()
        self.scaler.fit(np.reshape(self.time_series_values, (-1, 1)))
        scaled_series = self.scaler.transform(np.reshape(self.time_series_values, (-1, 1)))

        train_X = np.array(list(more_itertools.windowed(scaled_series[0:-1], n=train_window)))
        train_Y = np.reshape(scaled_series[train_window:], (-1, 1))
        test_X = np.reshape(scaled_series[len(scaled_series) - train_window:], (1, -1))

        return train_X, train_Y, test_X

    def predict(self, future_points: int = 1) -> List[Prediction]:

        predictions = []
        sigmas = []

        train_X, train_Y, test_X = self.build_train_test()
        train_X = np.squeeze(train_X,axis=2)

        k = gpflow.kernels.Linear(1)
        model = gpflow.models.GPR(train_X,train_Y,kern=k)

        opt = gpflow.train.ScipyOptimizer()
        opt.minimize(model)

        for k in range(future_points):

            mean, var = model.predict_y(test_X)
            result = self.scaler.inverse_transform(mean)[0][0]

            predictions.append(result)
            sigmas.append(var[0][0])

            self.add_observation(result)
            _, _, test_X = self.build_train_test()

        return [Prediction(predictions[i],sigmas[i]) for i in np.arange(len(predictions))]
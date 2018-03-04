from typing import List

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import more_itertools

from core.AbstractForecastingModel import AbstractForecastingModel
from core.Prediction import Prediction

train_window = 2

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'ls'}

class GradientBoosting_Directive(AbstractForecastingModel):
    @property
    def name(self) -> str:
        return "GradientBoosting_Directive"

    def build_train_test(self, future_points: int = 1):

        train_X = np.array(list(more_itertools.windowed(self.time_series_values[0:-future_points], n=train_window, step=1)))
        train_Y = np.reshape(self.time_series_values[train_window + future_points - 1:], (-1, 1))
        test_X = np.reshape(self.time_series_values[len(self.time_series_values) - train_window:], (1, -1))

        return train_X, train_Y, test_X

    def predict(self, future_points: int = 1) -> List[Prediction]:

        predictions = []

        for i in range(1, future_points + 1):

            train_X, train_Y, test_X = self.build_train_test(i)

            regr = GradientBoostingRegressor(**params)
            train_model = regr.fit(train_X, train_Y.ravel())

            result = train_model.predict(test_X)[0]
            predictions.append(result)

        return [Prediction(predictions[i]) for i in np.arange(len(predictions))]
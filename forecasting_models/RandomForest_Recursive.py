from typing import List

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import more_itertools

from core.AbstractForecastingModel import AbstractForecastingModel
from core.Prediction import Prediction

train_window = 5

class RandomForest_Recursive(AbstractForecastingModel):
    @property
    def name(self) -> str:
        return "RandomForest_Recursive"

    def build_train_test(self, train_window: int = 1):

        train_X = np.array(list(more_itertools.windowed(self.time_series_values[0:-1],n=train_window, step=1)))
        train_Y = np.reshape(self.time_series_values[train_window:],(-1,1))
        test_X = np.reshape(self.time_series_values[len(self.time_series_values)-train_window:],(1,-1))

        return train_X, train_Y, test_X

    def predict(self, future_points: int = 1) -> List[Prediction]:

        predictions = []

        train_X, train_Y, test_X = self.build_train_test(train_window)

        regr = RandomForestRegressor(max_depth=3)
        train_model = regr.fit(train_X, train_Y.ravel())

        for i in range(future_points):

            result = train_model.predict(test_X)[0]
            predictions.append(result)

            self.add_observation(result)
            _, _, test_X = self.build_train_test(train_window)

        return [Prediction(predictions[i]) for i in np.arange(len(predictions))]
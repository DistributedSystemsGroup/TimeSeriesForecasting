from typing import List

import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import more_itertools


from core.AbstractForecastingModel import AbstractForecastingModel
from core.Prediction import Prediction

train_window = 2

parameters = {"kernel": ["rbf"], "C": [10, 100, 1000, 10000], "gamma": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}

class Svr_Directive(AbstractForecastingModel):
    @property
    def name(self) -> str:
        return "Svr_Directive"

    def build_train_test(self, future_points: int = 1):

        train_X = np.array(list(more_itertools.windowed(self.time_series_values[0:-future_points], n=train_window, step=1)))
        train_Y = np.reshape(self.time_series_values[train_window+future_points-1:],(-1,1))
        test_X = np.reshape(self.time_series_values[len(self.time_series_values)-train_window:],(1,-1))

        return train_X, train_Y, test_X

    def predict(self, future_points: int = 1) -> List[Prediction]:

        predictions = []

        for i in range(1,future_points+1):

            train_X, train_Y, test_X = self.build_train_test(i)

            clf = GridSearchCV(SVR(), parameters)
            train_model = clf.fit(train_X, train_Y.ravel())

            result = train_model.predict(test_X)[0]
            predictions.append(result)

        return [Prediction(predictions[i]) for i in np.arange(len(predictions))]
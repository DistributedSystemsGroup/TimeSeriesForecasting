from typing import List

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

from core.AbstractRecursiveForecastingModel import AbstractRecursiveForecastingModel
from core.Prediction import Prediction

train_window = 2

parameters = {
    "kernel": ["rbf"],
    "C": [1, 10, 100, 1000, 10000, 100000],
    "gamma": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
}


class SvrRecursive(AbstractRecursiveForecastingModel):
    @property
    def name(self) -> str:
        return "SvrRecursive"

    def predict(self, future_points: int = 1) -> List[Prediction]:
        predictions = []

        train_x, train_y, test_x = self.build_train_test(train_window=train_window)

        clf = GridSearchCV(SVR(), parameters)
        train_model = clf.fit(train_x, train_y.ravel())

        for i in range(future_points):
            result = train_model.predict(test_x)[0]
            predictions.append(Prediction(result))

            self.add_observation(result)
            _, _, test_x = self.build_train_test(train_window=train_window)

        return predictions

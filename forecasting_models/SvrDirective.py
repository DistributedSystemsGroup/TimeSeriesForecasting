from typing import List

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

from core.AbstractDirectiveForecastingModel import AbstractDirectiveForecastingModel
from core.Prediction import Prediction

train_window = 2

parameters = {
    "kernel": ["rbf"],
    "C": [10, 100, 1000, 10000],
    "gamma": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}


class SvrDirective(AbstractDirectiveForecastingModel):
    @property
    def name(self) -> str:
        return "SvrDirective"

    def predict(self, future_points: int = 1) -> List[Prediction]:
        predictions = []

        for i in range(1, future_points + 1):
            train_x, train_y, test_x = self.build_train_test(future_points=i, train_window=train_window)

            clf = GridSearchCV(SVR(), parameters)
            train_model = clf.fit(train_x, train_y.ravel())

            result = train_model.predict(test_x)[0]
            predictions.append(Prediction(result))

        return predictions

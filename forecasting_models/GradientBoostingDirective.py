from typing import List

from sklearn.ensemble import GradientBoostingRegressor

from core.Prediction import Prediction
from core.AbstractDirectiveForecastingModel import AbstractDirectiveForecastingModel

train_window = 2

params = {
    'n_estimators': 500,
    'max_depth': 4,
    'min_samples_split': 2,
    'learning_rate': 0.01,
    'loss': 'ls'}


class GradientBoostingDirective(AbstractDirectiveForecastingModel):
    @property
    def name(self) -> str:
        return "GradientBoostingDirective"

    def predict(self, future_points: int = 1) -> List[Prediction]:
        predictions = []

        for i in range(1, future_points + 1):
            train_x, train_y, test_x = self.build_train_test(future_points=i, train_window=train_window)

            regr = GradientBoostingRegressor(**params)
            train_model = regr.fit(train_x, train_y.ravel())

            result = train_model.predict(test_x)[0]
            predictions.append(Prediction(result))

        return predictions

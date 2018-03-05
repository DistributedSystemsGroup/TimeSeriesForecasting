from typing import List

from sklearn.ensemble import RandomForestRegressor

from core.AbstractDirectiveForecastingModel import AbstractDirectiveForecastingModel
from core.Prediction import Prediction

train_window = 2


class RandomForestDirective(AbstractDirectiveForecastingModel):
    @property
    def name(self) -> str:
        return "RandomForestDirective"

    def predict(self, future_points: int = 1) -> List[Prediction]:
        predictions = []

        for i in range(1, future_points + 1):
            train_x, train_y, test_x = self.build_train_test(future_points=i, train_window=train_window)

            regr = RandomForestRegressor(max_depth=2)
            train_model = regr.fit(train_x, train_y.ravel())

            result = train_model.predict(test_x)[0]
            predictions.append(Prediction(result))

        return predictions

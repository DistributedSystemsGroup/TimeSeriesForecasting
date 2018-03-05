from typing import List

from sklearn.ensemble import RandomForestRegressor

from core.AbstractRecursiveForecastingModel import AbstractRecursiveForecastingModel
from core.Prediction import Prediction

train_window = 5


class RandomForestRecursive(AbstractRecursiveForecastingModel):
    @property
    def name(self) -> str:
        return "RandomForestRecursive"

    def predict(self, future_points: int = 1) -> List[Prediction]:
        predictions = []

        train_x, train_y, test_x = self.build_train_test(train_window=train_window)

        regr = RandomForestRegressor(max_depth=3)
        train_model = regr.fit(train_x, train_y.ravel())

        for i in range(future_points):
            result = train_model.predict(test_x)[0]
            predictions.append(Prediction(result))

            self.add_observation(result)
            _, _, test_x = self.build_train_test(train_window=train_window)

        return predictions

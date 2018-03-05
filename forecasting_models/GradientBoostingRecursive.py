from typing import List

from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.model_selection import GridSearchCV

from core.AbstractRecursiveForecastingModel import AbstractRecursiveForecastingModel
from core.Prediction import Prediction

train_window = 5

params = {
    'n_estimators': 500,
    'max_depth': 4,
    'min_samples_split': 2,
    'learning_rate': 0.01,
    'loss': 'ls'}


class GradientBoostingRecursive(AbstractRecursiveForecastingModel):
    @property
    def name(self) -> str:
        return "GradientBoostingRecursive"

    def predict(self, future_points: int = 1) -> List[Prediction]:
        predictions = []

        train_x, train_y, test_x = self.build_train_test(train_window=train_window)

        regr = GradientBoostingRegressor(**params)
        # regr = GridSearchCV(GradientBoostingRegressor(), params)
        train_model = regr.fit(train_x, train_y.ravel())

        for i in range(future_points):
            result = train_model.predict(test_x)[0]
            predictions.append(Prediction(result))

            self.add_observation(result)
            _, _, test_x = self.build_train_test(train_window=train_window)

        return predictions

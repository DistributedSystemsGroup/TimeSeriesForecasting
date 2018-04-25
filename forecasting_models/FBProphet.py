from typing import List

import numpy as np
import pandas as pd

from core.AbstractForecastingModel import AbstractForecastingModel
from core.Prediction import Prediction
from fbprophet import Prophet

np.random.seed(1234)

class FBProphet(AbstractForecastingModel):
    @property
    def name(self) -> str:
        return "FBProphet"

    def build_train_test(self, future_points: int = 1):

        train_X = [float(i) for i in list(range(len(self.time_series_values)))]
        train_Y = self.time_series_values
        test_X = [float(i) for i in list(range(len(self.time_series_values), len(self.time_series_values) + future_points))]

        return train_X, train_Y, test_X

    def build_df(self, train_X, train_Y):

        dict_table = {"ds": train_X, "y": train_Y}
        df = pd.DataFrame(dict_table)

        return df

    def predict(self, future_points: int = 1) -> List[Prediction]:

        train_X, train_Y, test_X = self.build_train_test(future_points)
        df = self.build_df(train_X, train_Y)

        model = Prophet(uncertainty_samples=2000)
        model.fit(df)

        future = pd.DataFrame({"ds": test_X})
        forecast = model.predictive_samples(future)

        posterior_pred = forecast["yhat"]
        pred_mean = np.mean(posterior_pred, axis=1)
        pred_var = np.var(posterior_pred, axis=1)

        return [Prediction(pred_mean[i],pred_var[i]) for i in np.arange(len(posterior_pred))]

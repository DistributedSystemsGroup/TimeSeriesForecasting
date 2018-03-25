from typing import List

import numpy as np
from nnet_ts.TimeSeriesNnet import TimeSeriesNnet

from core.AbstractDirectiveForecastingModel import AbstractDirectiveForecastingModel
from core.Prediction import Prediction

train_window = 2
np.random.seed(1234)

class NeuralNetwork(AbstractDirectiveForecastingModel):
    @property
    def name(self) -> str:
        return "NeuralNetwork"

    def predict(self, future_points: int = 1) -> List[Prediction]:

        neural_net = TimeSeriesNnet(hidden_layers=[50,20], activation_functions=['sigmoid','sigmoid'])

        neural_net.fit(np.array(self.time_series_values), lag=train_window, epochs=1000, verbose=0)
        predictions = neural_net.predict_ahead(n_ahead=future_points)

        return [Prediction(predictions[i]) for i in np.arange(len(predictions))]
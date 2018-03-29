from typing import List

import numpy as np
import more_itertools
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers

from core.AbstractForecastingModel import AbstractForecastingModel
from core.Prediction import Prediction

n_hidden = 20
#n_layers = 2
training_iters = 150
#learning_rate = 0.1
#num_features = 1
train_window = 7

np.random.seed(1234)

class Lstm_Keras(AbstractForecastingModel):
    @property
    def name(self) -> str:
        return "Lstm_Keras"

    def build_train_test(self):

        self.scaler = StandardScaler()
        self.scaler.fit(np.reshape(self.time_series_values, (-1, 1)))
        scaled_series = self.scaler.transform(np.reshape(self.time_series_values, (-1, 1)))

        train_X = np.array(list(more_itertools.windowed(scaled_series[0:-1], n=train_window)))
        train_Y = np.reshape(scaled_series[train_window:], (-1, 1))
        test_X = np.reshape(scaled_series[len(scaled_series) - train_window:], (1, -1))

        return train_X, train_Y, test_X

    def predict(self, future_points: int = 1) -> List[Prediction]:

        predictions = []

        train_X, train_Y, test_X = self.build_train_test()

        train_X = np.transpose(train_X,(0,2,1))

        model = Sequential()
        model.add(LSTM(n_hidden, input_shape=(1, train_window)))
        model.add(Dense(1))

        optimizer = optimizers.Adam(lr=0.001)
        model.compile(loss='mean_squared_error', optimizer=optimizer)

        model.fit(train_X, train_Y, epochs=training_iters, batch_size=1, verbose=1)

        for k in range(future_points):

            test_X = np.expand_dims(test_X, axis=2).transpose(0,2,1)
            result = model.predict(test_X)
            result = self.scaler.inverse_transform(result)

            predictions.append(result[0][0])

            self.add_observation(result)
            _, _, test_X = self.build_train_test()

        return [Prediction(predictions[i]) for i in np.arange(len(predictions))]
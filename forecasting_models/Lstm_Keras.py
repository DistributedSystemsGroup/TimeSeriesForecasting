from typing import List

import numpy as np
import more_itertools
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers
from keras import backend as K
import tensorflow as tf

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

    _current_model = None
    scaler = None

    @property
    def current_model(self) -> Sequential:
        if self._current_model == None:
            self._current_model = Sequential()
            self._current_model.add(LSTM(n_hidden, input_shape=(1, train_window)))
            self._current_model.add(Dense(1))
            optimizer = optimizers.Adam(lr=0.001)
            self._current_model.compile(loss='mean_squared_error', optimizer=optimizer)
        return self._current_model

    def build_train_test(self, build_test_x_only=False):
        # we only allocate object "scaler" once
        if not self.scaler:
            self.scaler = StandardScaler()

        time_series_values_reshaped = np.reshape(self.time_series_values, (-1, 1))
        self.scaler.partial_fit(time_series_values_reshaped)

        if build_test_x_only:
            # only scale the values for test_X
            scaled_series = self.scaler.transform(time_series_values_reshaped[len(time_series_values_reshaped) - train_window:])
            test_X = np.reshape(scaled_series, (1, -1))
            return None, None, test_X
        else:
            # we scale the whole series
            scaled_series = self.scaler.transform(time_series_values_reshaped)
            test_X = np.reshape(scaled_series[len(scaled_series) - train_window:], (1, -1))
            train_X = np.array(list(more_itertools.windowed(scaled_series[0:-1], n=train_window)))
            train_Y = np.reshape(scaled_series[train_window:], (-1, 1))

            return train_X, train_Y, test_X

    def predict(self, future_points: int = 1) -> List[Prediction]:

        predictions = []

        train_X, train_Y, test_X = self.build_train_test()

        train_X = np.transpose(train_X,(0,2,1))

        # model = Sequential()
        # model.add(LSTM(n_hidden, input_shape=(1, train_window)))
        # model.add(Dense(1))
        model = self.current_model

        model.fit(train_X, train_Y, epochs=training_iters, batch_size=10, verbose=0)

        for k in range(future_points):

            test_X = np.expand_dims(test_X, axis=2).transpose(0,2,1)
            result = model.predict(test_X)

            result = self.scaler.inverse_transform(result)

            predictions.append(result[0][0])

            self.add_observation(result)

            _, _, test_X = self.build_train_test(build_test_x_only=True)

        # K.clear_session()
        # tf.reset_default_graph()

        return [Prediction(predictions[i]) for i in np.arange(len(predictions))]
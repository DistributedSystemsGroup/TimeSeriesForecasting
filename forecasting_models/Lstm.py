from typing import List

import numpy as np
import more_itertools
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.contrib import rnn

from core.AbstractForecastingModel import AbstractForecastingModel
from core.Prediction import Prediction

n_hidden = 40
n_layers = 1
training_iters = 150
learning_rate = 0.001
num_features = 1
train_window = 2

tf.set_random_seed(1234)

class Lstm(AbstractForecastingModel):

    @property
    def name(self) -> str:
        return "Lstm"

    def build_train_test(self):

        self.scaler = StandardScaler()
        self.scaler.fit(np.reshape(self.time_series_values, (-1,1)))
        scaled_series = self.scaler.transform(np.reshape(self.time_series_values, (-1,1)))

        train_X = np.array(list(more_itertools.windowed(scaled_series[0:-1], n=train_window)))
        train_Y = np.reshape(scaled_series[train_window:], (-1, 1))
        test_X = np.reshape(scaled_series[len(scaled_series)-train_window:],(1,-1))

        return train_X, train_Y, test_X

    def make_cell(self):
        return rnn.BasicLSTMCell(n_hidden, state_is_tuple=True)

    def build_RNN(self, x, weights, biases):

        rnn_cell = rnn.MultiRNNCell([self.make_cell() for _ in range(n_layers)], state_is_tuple=True)
        outputs, _ = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

        output = tf.matmul(outputs[-1], weights) + biases

        return output


    def predict(self, future_points: int = 1) -> List[Prediction]:

        tf.reset_default_graph()

        train_X, train_Y, test_X = self.build_train_test()

        X = tf.placeholder(tf.float32, [None, train_window, num_features],name="X")
        y = tf.placeholder(tf.float32, [None, num_features],name="y")

        weigths = tf.Variable(tf.random_normal([n_hidden, num_features]))
        biases = tf.Variable(tf.random_normal([num_features]))

        input = tf.unstack(X,train_window,1)

        rnn_output = self.build_RNN(input,weigths,biases)

        loss = tf.losses.mean_squared_error(rnn_output,y)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

        predictions = []

        with tf.Session() as sess:

            sess.run(init)

            for i in range(training_iters):
                sess.run(train_op, feed_dict={X: train_X, y: train_Y})

            for k in range(future_points):

                test_X = np.expand_dims(test_X, axis=2)
                result = sess.run(rnn_output, feed_dict={X: test_X})
                result = self.scaler.inverse_transform(result)

                predictions.append(result[0][0])

                self.add_observation(result)
                _, _, test_X = self.build_train_test()

        return [Prediction(predictions[i]) for i in np.arange(len(predictions))]
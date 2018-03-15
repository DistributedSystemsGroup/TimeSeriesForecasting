import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import more_itertools
import numpy as np

from core.AbstractForecastingModel import AbstractForecastingModel
from core.Prediction import Prediction

input_size = 1
num_features = 1
train_window = 3
n_hidden = 50
n_layers = 1
training_iters = 150
learning_rate = 0.1

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self,x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out

class Lstm(AbstractForecastingModel):
    @property
    def name(self) -> str:
        return "Lstm_PyTorch"

    def build_train_test(self):

        train_X = np.array(list(more_itertools.windowed(self.time_series_values[0:-1], n=train_window)))
        train_Y = np.reshape(self.time_series_values[train_window:], (-1, 1))
        test_X = np.reshape(self.time_series_values[len(self.time_series_values)-train_window:],(1,-1))

        return train_X, train_Y, test_X


    def predict(self, future_points: int = 1):

        train_X, train_Y, test_X = self.build_train_test()

        rnn = RNN(input_size, n_hidden, n_layers, num_features)

        # Loss and Optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)

        # Train the model
        for epoch in range(training_iters):
            train_X_tens = torch.from_numpy(train_X)
            train_Y_tens = torch.from_numpy(train_Y)
            train_inputs = Variable(train_X_tens.view(-1,train_window,input_size)).float()
            train_labels = Variable(train_Y_tens).float()

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = rnn(train_inputs)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()

        # Test the model
        predictions = []
        for i in range(future_points):
            test_X_tens = torch.from_numpy(test_X)
            test_inputs = Variable(test_X_tens.view(-1,train_window,input_size)).float()
            outputs = rnn(test_inputs)
            result = outputs.data.numpy()[0][0]
            predictions.append(result)

            self.add_observation(result)
            _, _, test_X = self.build_train_test()

        return [Prediction(predictions[i]) for i in np.arange(len(predictions))]

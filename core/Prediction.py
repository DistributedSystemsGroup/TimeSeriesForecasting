import numpy as np


class Prediction:
    def __init__(self, value: float, variance: float = 0.0):
        self.value = value
        self.variance = variance
        self.stddev = np.sqrt(variance)

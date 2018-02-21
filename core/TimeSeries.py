from typing import List


class TimeSeries:
    def __init__(self, values: List[float], forecasting_window: int = 1, minimum_observations: int = 1):
        self.values = values
        self.forecasting_window = forecasting_window
        self.minimum_observations = minimum_observations

    def get_values(self) -> List[float]:
        return self.values

    def get_forecasting_window(self) -> int:
        return self.forecasting_window

    def get_minimum_observations(self) -> int:
        return self.minimum_observations

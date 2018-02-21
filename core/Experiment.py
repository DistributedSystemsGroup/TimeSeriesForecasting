import csv
from datetime import datetime
from typing import List

import numpy as np

import os

from core.AbstractForecastingModel import AbstractForecastingModel
from core.TimeSeries import TimeSeries

date_format = "%Y-%m-%d-%H-%M-%S"


class Experiment:

    def __init__(self, model: AbstractForecastingModel, time_series: List[TimeSeries], ):
        self.model = model
        self.time_series = time_series

    def run(self):
        directory_path = os.path.join("experiment_results", datetime.now().strftime(date_format))
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        with open(os.path.join(directory_path, "{}_predicted.csv".format(self.model.name)), "w", newline='') as csvfile:
            fields_names = ["values", "stddev"]
            csv_writer = csv.DictWriter(csvfile, fieldnames=fields_names,
                                        delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writeheader()

            for ts in self.time_series:
                w = ts.forecasting_window
                i = ts.minimum_observations

                predicted_values = ts.values[:i]
                predicted_stddev = [0] * ts.minimum_observations

                self.model.add_observations(ts.values[:i])

                while i < len(ts.values):
                    predictions = self.model.predict(w)
                    assert w == len(predictions), \
                        "The returned predictions are less than what requested. {} vs {}".format(len(predictions), w)
                    for p in predictions:
                        predicted_values.append(p.value)
                        predicted_stddev.append(np.sqrt(p.variance))
                    self.model.add_observations(ts.values[i:i + w])
                    i += w

                csv_writer.writerow({
                    "values": " ".join(str(x) for x in predicted_values),
                    "stddev": " ".join(str(x) for x in predicted_stddev)
                })



import csv
from datetime import datetime
from typing import List

import numpy as np

import os

from shutil import copyfile

from core.AbstractForecastingModel import AbstractForecastingModel
from core.TimeSeries import TimeSeries

date_format = "%Y-%m-%d-%H-%M-%S"


class Experiment:

    def __init__(self, input_file_path: str, model: AbstractForecastingModel, time_series: List[TimeSeries], ):
        self.model = model
        self.time_series = time_series
        self.input_file_path = input_file_path

    def run(self):
        directory_path = os.path.join("experiment_results", datetime.now().strftime(date_format))
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # input_filename = os.path.basename(self.input_file_path)
        # copyfile(self.input_file_path, os.path.join(directory_path, input_filename))

        # with open(os.path.join(directory_path, "{}_predicted_{}.csv".format(input_filename, self.model.name)),
        with open(os.path.join(directory_path, "predicted_{}.csv".format(self.model.name)),
                  "w", newline='') as csvfile:
            fields_names = ["observed", "predicted", "stddev"]
            csv_writer = csv.DictWriter(csvfile, fieldnames=fields_names,
                                        delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writeheader()

            for ts in self.time_series:
                self.model.reset()

                predicted_values = ts.values[:ts.minimum_observations]
                predicted_stddev = [0] * ts.minimum_observations

                i = 0
                while i < len(ts.values):
                    value = ts.values[i]
                    if len(self.model.get_observations()) >= ts.minimum_observations:
                        predictions = self.model.predict(ts.forecasting_window)
                        assert ts.forecasting_window == len(predictions), \
                            "The returned predictions are less than what requested. {} vs {}".format(
                                len(predictions), ts.forecasting_window)
                        for p in predictions:
                            predicted_values.append(p.value)
                            predicted_stddev.append(np.sqrt(p.variance))
                        if len(predicted_values) >= len(ts.values):
                            break
                    self.model.add_observation(value)
                    i += 1

                csv_writer.writerow({
                    "observed": " ".join(str(x) for x in ts.values),
                    "predicted": " ".join(str(x) for x in predicted_values),
                    "stddev": " ".join(str(x) for x in predicted_stddev)
                })



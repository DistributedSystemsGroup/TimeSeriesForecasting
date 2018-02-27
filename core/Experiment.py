import csv
from datetime import datetime
from typing import List

import numpy as np

import os

from shutil import copyfile

from core.AbstractForecastingModel import AbstractForecastingModel
from core.Prediction import Prediction
from core.TimeSeries import TimeSeries

date_format = "%Y-%m-%d-%H-%M-%S"
EXPERIMENTS_FOLDER_NAME = "experiment_results"


class Experiment:

    def __init__(self, model: AbstractForecastingModel, time_series: List[TimeSeries], ):
        self.model = model
        self.time_series = time_series
        self.csv_writer = None

        experiment_directory_path = os.path.join(EXPERIMENTS_FOLDER_NAME, datetime.now().strftime(date_format))
        if not os.path.exists(experiment_directory_path):
            os.makedirs(experiment_directory_path)

        self.experiment_result_file_path = \
            os.path.join(experiment_directory_path, "predicted_{}.csv".format(self.model.name))

    def __dump_result_on_csv__(self, csv_file, row_to_write):
        if self.csv_writer is None:
            self.csv_writer = csv.DictWriter(csv_file, fieldnames=row_to_write.keys(),
                                             delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            self.csv_writer.writeheader()
        self.csv_writer.writerow(row_to_write)

    def run(self):
        with open(self.experiment_result_file_path, "w", newline='') as csv_file:
            for ts in self.time_series:
                assert len(ts.predictions) == 0, \
                    "There are already some predictions ({}) inside this TimeSeries object.".format(len(ts.predictions))
                self.model.reset()

                i = 0
                while i < ts.minimum_observations:
                    ts.predictions.append(Prediction(ts.observations[i], 0))
                    i += 1

                i = 0
                while i < len(ts.observations):
                    value = ts.observations[i]
                    if len(self.model.get_observations()) >= ts.minimum_observations:
                        predictions = self.model.predict(ts.forecasting_window)
                        assert ts.forecasting_window == len(predictions), \
                            "The returned predictions are less than what requested. {} vs {}".format(
                                len(predictions), ts.forecasting_window)
                        ts.predictions.extend(predictions)
                        if len(ts.predictions) >= len(ts.observations):
                            break
                    self.model.add_observation(value)
                    i += 1

                self.__dump_result_on_csv__(csv_file, ts.to_csv())

import csv
import logging
import logging.config
from datetime import datetime
from typing import List

import os

from core.AbstractForecastingModel import AbstractForecastingModel
from core.Prediction import Prediction
from core.TimeSeries import TimeSeries
from utils.MultiProcessLogger import MultiProcessLogger

date_format = "%Y-%m-%d-%H-%M-%S"
EXPERIMENTS_FOLDER_NAME = "experiment_results"


class Experiment:

    def __init__(self, model: AbstractForecastingModel, time_series: List[TimeSeries], csv_writing: bool=True):
        self.model = model

        self.time_series = time_series

        self.csv_writing = csv_writing
        if self.csv_writing:
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

    def run(self, mp_logger: MultiProcessLogger=None):
        if mp_logger is not None:
            mp_logger.add_logger()
            logger = MultiProcessLogger.logger(self.model.name)
            logger.info("Experiment Started.")
        else:
            logger = None

        if self.csv_writing:
            csv_file = open(self.experiment_result_file_path, "w", newline='')
        else:
            csv_file = None

        if logger is not None:
            logger.debug(" Predicting {} time-series.".format(len(self.time_series)))

        for ts in self.time_series:
            self.model.reset()
            ts.reset()
            assert len(ts.predictions) == 0, \
                "There are already some predictions ({}) inside this TimeSeries object.".format(len(ts.predictions))

            if logger is not None:
                logger.debug("  Time-Series of {} observations.".format(len(ts.observations)))

            i = 0
            while i < ts.minimum_observations:
                ts.predictions.append(Prediction(ts.observations[i], 0))
                i += 1

            i = 0
            while i < len(ts.observations):
                value = ts.observations[i]
                if len(self.model.get_observations()) >= ts.minimum_observations:
                    # if logger is not None:
                    #     logger.debug("   Predicting the points from {} to {}.".format(i, i + ts.forecasting_window))
                    predictions = self.model.predict(ts.forecasting_window)
                    assert ts.forecasting_window == len(predictions), \
                        "The returned predictions are less than what requested. {} vs {}".format(
                            len(predictions), ts.forecasting_window)
                    ts.predictions.extend(predictions)
                    if len(ts.predictions) >= len(ts.observations):
                        break
                self.model.add_observation(value)
                i += 1
            if len(ts.predictions) >= len(ts.observations):
                ts.predictions = ts.predictions[:len(ts.observations)]
             
            if csv_file is not None:
                self.__dump_result_on_csv__(csv_file, ts.to_csv())

        if csv_file is not None:
            csv_file.close()

        if logger is not None:
            logger.info("Experiment Finished.")

        return self.time_series

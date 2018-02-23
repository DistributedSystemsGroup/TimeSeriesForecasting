import csv
import logging

import sys
from multiprocessing.pool import Pool

import os

from core.Experiment import Experiment
from core.TimeSeries import TimeSeries
from forecasting_models.Arima import Arima
from forecasting_models.DummyPrevious import DummyPrevious


def run_experiment(exp: Experiment):
    exp.run()


if __name__ == '__main__':
    def set_csv_field_size_limit():
        max_int = sys.maxsize

        decrement = True
        while decrement:
            # decrease the maxInt value by factor 10
            # as long as the OverflowError occurs.

            decrement = False
            try:
                csv.field_size_limit(max_int)
            except OverflowError:
                max_int = int(max_int / 10)
                decrement = True


    number_of_processes = 1
    input_filename = "M3C.csv"
    # input_filename = "google-tss_1.csv"
    input_file_path = os.path.join("traces", input_filename)

    set_csv_field_size_limit()

    tss = []
    with open(input_file_path) as csvfile:
        reader = csv.DictReader(csvfile)
        if "values" not in reader.fieldnames:
            RuntimeError("No columns named 'values' inside the provided csv!")
            sys.exit(-1)

        for row in reader:
            observations = [float(x) for x in row.pop("values").split(" ")]
            tss.append(TimeSeries(observations, **row))

    if len(tss) > 0:
        experiments = [
            Experiment(DummyPrevious(), tss),
            # Experiment(input_file_path, Arima(), tss)
        ]

        with Pool(processes=number_of_processes) as p:
            p.map(run_experiment, experiments)

    else:
        logging.warning("No Time Series to process.")

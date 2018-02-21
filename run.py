import csv
import logging

import sys
from multiprocessing.pool import Pool

import os

from core.Experiment import Experiment
from core.TimeSeries import TimeSeries
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

    set_csv_field_size_limit()

    tss = []
    with open(os.path.join("traces", "google-tss_10.csv")) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ts = None
            # FIXME: Very ugly, let's think at a better way create a new TimeSeries object
            if "values" in row:
                values = [float(x) for x in row["values"].split(" ")]
                if "forecasting_window" in row:
                    if "minimum_observations" in row:
                        ts = TimeSeries(values, int(row["forecasting_window"]), int(row["minimum_observations"]))
                    else:
                        ts = TimeSeries(values, int(row["forecasting_window"]))
                else:
                    ts = TimeSeries(values)
            else:
                RuntimeError("No columns named 'values' inside the provided csv!")
                sys.exit(-1)

            tss.append(ts)

    if len(tss) > 0:
        model_to_use = DummyPrevious()

        experiments = [Experiment(model_to_use, tss)]

        with Pool(processes=number_of_processes) as p:
            p.map(run_experiment, experiments)

    else:
        logging.warning("No Time Series to process.")

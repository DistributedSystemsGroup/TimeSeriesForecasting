import csv
import logging

import sys
from multiprocessing.pool import Pool

import os

from core.Experiment import Experiment
from core.TimeSeries import TimeSeries
from forecasting_models.Arima import Arima
from forecasting_models.DummyPrevious import DummyPrevious
from forecasting_models.ExpSmoothing_NoTrend import ExpSmoothing_NoTrend
from forecasting_models.ExpSmoothing_Trend import ExpSmoothing_Trend
from forecasting_models.Svr_Recursive import Svr_Recursive
from forecasting_models.Svr_Directive import Svr_Directive
from forecasting_models.RandomForest_Recursive import RandomForest_Recursive
from forecasting_models.RandomForest_Directive import RandomForest_Directive
from forecasting_models.GradientBoosting_Recursive import GradientBoosting_Recursive
from forecasting_models.GradientBoosting_Directive import GradientBoosting_Directive


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
        tss.extend([TimeSeries(observations=[float(x) for x in row.pop("values").split(" ")], **row) for row in reader])

    if len(tss) > 0:
        experiments = [
            Experiment(DummyPrevious(), tss),
            #Experiment(input_file_path, Arima(), tss)
            #Experiment(ExpSmoothing_NoTrend(), tss)
            #Experiment(ExpSmoothing_Trend(), tss)
            #Experiment(Svr_Recursive(), tss)
            #Experiment(Svr_Directive(), tss)
            #Experiment(RandomForest_Recursive(), tss)
            #Experiment(RandomForest_Directive(), tss)
            #Experiment(GradientBoosting_Recursive(), tss)
            #Experiment(GradientBoosting_Directive(), tss)
        ]

        with Pool(processes=number_of_processes) as p:
            p.map(run_experiment, experiments)

    else:
        logging.warning("No Time Series to process.")

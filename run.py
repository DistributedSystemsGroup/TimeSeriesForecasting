import csv
import argparse

import sys
from multiprocessing.pool import Pool

import os

from multiprocessing import Queue

from core.Experiment import Experiment
from core.TimeSeries import TimeSeries
from forecasting_models.Arima import Arima
from forecasting_models.AutoArima import AutoArima
from forecasting_models.DummyPrevious import DummyPrevious
from forecasting_models.ExpSmoothing import ExpSmoothing
from forecasting_models.GradientBoostingDirective import GradientBoostingDirective
from forecasting_models.GradientBoostingRecursive import GradientBoostingRecursive
from forecasting_models.RandomForestDirective import RandomForestDirective
from forecasting_models.RandomForestRecursive import RandomForestRecursive
from forecasting_models.SvrDirective import SvrDirective
from forecasting_models.SvrRecursive import SvrRecursive

from utils.MultiProcessLogger import MultiProcessLogger
from utils.utils import set_csv_field_size_limit


def run_experiment(exp: Experiment):
    exp.run(multiple_process_logger)


multiple_process_logger = MultiProcessLogger(os.path.join("logging.conf"), Queue())

if __name__ == '__main__':
    multiple_process_logger.start()
    logger = MultiProcessLogger.logger(__name__)

    parser = argparse.ArgumentParser(description="Experiment different TimeSeries Forecasting Models.")
    parser.add_argument(action='store', dest='input_trace_file', help='Input Trace File')
    parser.add_argument("-p", "--parallelism", type=int, default=1,
                        action='store', dest='parallelism', help='Parallelism')
    args = parser.parse_args()

    input_file_path = os.path.abspath(args.input_trace_file)

    set_csv_field_size_limit()

    logger.info("Loading file: {}".format(input_file_path))
    tss = []
    with open(input_file_path) as csvfile:
        reader = csv.DictReader(csvfile)
        if "values" not in reader.fieldnames:
            RuntimeError("No columns named 'values' inside the provided csv!")
            sys.exit(-1)
        tss.extend([TimeSeries(observations=[float(x) for x in row.pop("values").split(" ")], **row) for row in reader])
    logger.info("Loaded {} TimeSeries".format(len(tss)))

    if len(tss) > 0:
        models_to_test = [
            # DummyPrevious(),
            # Arima(),
            AutoArima(20),
            # ExpSmoothing(),
            # SvrRecursive(),
            # SvrDirective(),
            # RandomForestRecursive(),
            # RandomForestDirective(),
            # GradientBoostingRecursive(),
            # GradientBoostingDirective(),
        ]
        logger.info("Launching {} experiments with a parallelism of {}".format(len(models_to_test), args.parallelism))

        with Pool(processes=args.parallelism) as p:
            p.map(run_experiment, [Experiment(model, tss) for model in models_to_test])

        logger.info("All Experiments finished.")
        multiple_process_logger.stop()

    else:
        logger.warning("No Time Series to process.")

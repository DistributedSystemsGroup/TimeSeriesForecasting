import csv
import argparse
from datetime import datetime

import sys
from pyspark import SparkContext

import os

import logging.config

from core.TimeSeries import TimeSeries
from core.Experiment import Experiment

from forecasting_models.DummyPrevious import DummyPrevious
# from forecasting_models.Arima import Arima
from forecasting_models.AutoArima import AutoArima
# from forecasting_models.ExpSmoothing import ExpSmoothing
from forecasting_models.GradientBoostingDirective import GradientBoostingDirective
from forecasting_models.GradientBoostingRecursive import GradientBoostingRecursive
from forecasting_models.RandomForestDirective import RandomForestDirective
from forecasting_models.RandomForestRecursive import RandomForestRecursive
from forecasting_models.SvrDirective import SvrDirective
from forecasting_models.SvrRecursive import SvrRecursive

from utils.utils import set_csv_field_size_limit


def run_experiment(_v):
    return _v[1].name, Experiment(model=_v[1], time_series=[_v[0]], csv_writing=False).run()


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    logging.config.fileConfig(os.path.join(script_dir, "logging.conf"))
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Experiment different TimeSeries Forecasting Models.")
    parser.add_argument(action='store', nargs='+', dest='input_trace_files', help='Input Trace File')
    parser.add_argument("-p", "--parallelism", type=int, default=1,
                        action='store', dest='parallelism', help='Parallelism')
    args = parser.parse_args()

    set_csv_field_size_limit()

    tss = []

    for input_trace_file in args.input_trace_files:
        input_file_path = os.path.abspath(input_trace_file)
        logger.info("Loading file: {}".format(input_file_path))
        with open(input_file_path) as csvfile:
            reader = csv.DictReader(csvfile)
            if "values" not in reader.fieldnames:
                RuntimeError("No columns named 'values' inside the provided csv!")
                sys.exit(-1)
            tss.extend([TimeSeries(observations=[float(x) for x in row.pop("values").split(" ")], **row) for row in reader])
    logger.info("Loaded {} TimeSeries".format(len(tss)))

    if len(tss) > 0:
        models_to_test = [
            DummyPrevious(20),
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

        sc = SparkContext(appName="TimeSeriesForecasting")
        tss_rdd = sc.parallelize(tss, 48)
        models_rdd = sc.parallelize(models_to_test, len(models_to_test))
        # tss_rdd = sc.parallelize(list(range(100)), 1)

        logger.info(
            "Partitions -> Models: {} TSS: {}".format(models_rdd.getNumPartitions(), tss_rdd.getNumPartitions()))

        res_rdd = tss_rdd.cartesian(models_rdd).map(run_experiment).groupByKey().zipWithIndex().cache()

        date_format = "%Y-%m-%d-%H-%M-%S"
        experiment_directory_path = os.path.join(script_dir, "experiment_results",
                                                 datetime.now().strftime(date_format))
        if not os.path.exists(experiment_directory_path):
            os.makedirs(experiment_directory_path)

        for i in range(res_rdd.count()):
            ((model_name, _tss), _) = res_rdd.filter(lambda x: x[1] == i).collect()[0]

            csv_writer = None
            file_path = os.path.join(experiment_directory_path, "predicted_{}.csv".format(model_name))
            with open(file_path, "w", newline='') as csv_file:
                for _v in _tss:
                    row_to_write = _v[0].to_csv()
                    if csv_writer is None:
                        csv_writer = csv.DictWriter(csv_file, fieldnames=row_to_write.keys(),
                                                    delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writeheader()
                    csv_writer.writerow(row_to_write)

        sc.stop()
        logger.info("All Experiments finished. Results are in {}".format(experiment_directory_path))
    else:
        logger.warning("No Time Series to process.")

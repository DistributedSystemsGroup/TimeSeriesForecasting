import csv
import argparse
from datetime import datetime

import sys
from pyspark.storagelevel import StorageLevel
from pyspark import SparkContext


import os

import logging.config

from core.TimeSeries import TimeSeries
from core.Experiment import Experiment

from forecasting_models.DummyPrevious import DummyPrevious
# from forecasting_models.Arima import Arima
from forecasting_models.AutoArima import AutoArima
from forecasting_models.ExpSmoothing import ExpSmoothing
# from forecasting_models.FBProphet import FBProphet
from forecasting_models.GPRegression import GPRegression
from forecasting_models.GradientBoostingDirective import GradientBoostingDirective
from forecasting_models.GradientBoostingRecursive import GradientBoostingRecursive
from forecasting_models.Lstm_Keras import Lstm_Keras
from forecasting_models.NeuralNetwork import NeuralNetwork
from forecasting_models.RandomForestDirective import RandomForestDirective
from forecasting_models.RandomForestRecursive import RandomForestRecursive
from forecasting_models.Revarb import Revarb
from forecasting_models.SvrDirective import SvrDirective
from forecasting_models.SvrRecursive import SvrRecursive

from utils.utils import set_csv_field_size_limit

if os.path.exists('TimeSeriesForecasting.zip'):
    sys.path.insert(0, 'TimeSeriesForecasting.zip')
else:
    sys.path.insert(0, './TimeSeriesForecasting')


MAXIMUM_OBSERVATIONS = 400
HISTORY_SIZE = 10


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

    # Models that we want to test
    models_to_test = [
        DummyPrevious(),
        AutoArima(),
        ExpSmoothing(),
        SvrRecursive(),
        RandomForestRecursive(),
        GradientBoostingRecursive(),
        Lstm_Keras(),
        NeuralNetwork(),
        GPRegression(),
        Revarb(),
        # FBProphet()
    ]

    def parse_line_to_ts(line):        
        cols = line.split(",")
        if len(cols) < 3:
            return None

        # assume that the csv has a fixed schema: values,forecasting_window,minimum_observations
        # (if not, we can use a dictionary (col -> index) to load the data. However, in our experiments, the schema is fixed)
        # extract the first field and convert into series
        observations = [float(v) for v in cols[0].split(" ")]
        ts = TimeSeries(
            **{
                "observations": observations,
                "forecasting_window": int(cols[1]),
                "minimum_observations" : int(cols[2])
            }
        )

        if HISTORY_SIZE is not None and HISTORY_SIZE > 0:
            ts.minimum_observations = HISTORY_SIZE

        if MAXIMUM_OBSERVATIONS > 0:
            ts.observations = ts.observations[:MAXIMUM_OBSERVATIONS]

        if len(ts.observations) <= ts.minimum_observations:
            logger.warning("This TS has fewer points ({}) compared with the minimum observations {}"
                            .format(len(ts.observations), ts.minimum_observations))
            return None
        return ts

    # for each timeseries, train it with multiple models
    def series_to_models(ts):
        #result = []
        for model in models_to_test:
            try:
                r = Experiment(model=model, time_series=[ts], csv_writing=False).run()
                yield (model.name, r)
            except Exception as e:
                # we should log the error here
                logger.error("Error when building model {}".format(e))
                # throw e

    sc = SparkContext(appName="TimeSeriesForecasting")
    # sc.addPyFile("TimeSeriesForecasting.zip")

    print(args)

    tss_rdd = sc.textFile(",".join(args.input_trace_files), 60)
    header = tss_rdd.first()
    tss_rdd = tss_rdd.filter(lambda line: line != header) \
        .map(parse_line_to_ts) \
        .filter(lambda x: x) \
        .repartition(tss_rdd.context.defaultParallelism) \
        .persist(StorageLevel.MEMORY_AND_DISK)

    # force the execution
    # we want to convert all strings into number to save the memory before doing any computation
    logger.info("Number of timeseries:{}".format(tss_rdd.count()))
    
    # foreach series, build multiple models
    res_rdd = tss_rdd.flatMap(series_to_models).groupByKey().zipWithIndex().cache()

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

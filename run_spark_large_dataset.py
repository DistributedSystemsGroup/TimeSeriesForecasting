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

import math

from utils.utils import set_csv_field_size_limit

if os.path.exists('TimeSeriesForecasting.zip'):
    sys.path.insert(0, 'TimeSeriesForecasting.zip')
else:
    sys.path.insert(0, './TimeSeriesForecasting')


MAXIMUM_OBSERVATIONS = 400
HISTORY_SIZE = 10

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


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
        """Parse a line into a Series object"""     
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
        """Train multiple models using a single timeseries"""
        #result = []
        for model in models_to_test:
            try:
                r = Experiment(model=model, time_series=[ts], csv_writing=False).run()
                yield (model.name, r)
            except Exception as e:
                # we should log the error here
                logger.error("Error when building model {}".format(e))
                # throw e
    
    # only build a model for a series
    def series_to_model(ts, b_model):
        """Train a model using a timeseries"""
        model = b_model.value
        try:
            r = Experiment(model=model, time_series=[ts], csv_writing=False).run()
            return r
        except Exception as e:
            logger.error("Error when building model {}".format(e))
        return None

    sc = SparkContext(appName="TimeSeriesForecasting")
    sclogger = sc._jvm.org.apache.log4j
    sclogger.LogManager.getLogger("org"). setLevel( sclogger.Level.ERROR )
    sclogger.LogManager.getLogger("akka").setLevel( sclogger.Level.ERROR )

    print(args)

    tss_rdd = sc.textFile(",".join(args.input_trace_files), 60)
    header = tss_rdd.first()
    tss_rdd = tss_rdd.filter(lambda line: line != header) \
        .map(parse_line_to_ts) \
        .filter(lambda x: x) \
        .repartition(tss_rdd.context.defaultParallelism*2) \
        .persist(StorageLevel.MEMORY_AND_DISK)

    # force the execution
    # we want to convert all strings into number to save the memory before doing any computation
    logger.info("Number of timeseries:{}".format(tss_rdd.count()))

    date_format = "%Y-%m-%d-%H-%M-%S"
    experiment_directory_path = os.path.join(script_dir, "experiment_results",
                                                datetime.now().strftime(date_format))
    if not os.path.exists(experiment_directory_path):
        os.makedirs(experiment_directory_path)
    
    MAX_RESULT_COLLECT_IN_MB = 512 # MB

    for model in models_to_test:
        logger.info("Build {} models".format(model.name))
        b_model = sc.broadcast(model)

        # note that zipWithIndex is a transformation, however, it triggers a job
        # so, we cache our rdd here before calling 'zipWithIndex'
        models_rdd_wo_idx = tss_rdd.map(lambda s: series_to_model(s, b_model))\
                        .filter(lambda x: x)\
                        .persist(StorageLevel.MEMORY_AND_DISK)
                        
        models_rdd = models_rdd_wo_idx.zipWithIndex()\
                        .persist(StorageLevel.MEMORY_AND_DISK)

        # we try to calculate how many entries should be collected heuristically
        n_models_rdd = models_rdd.count()
        #sample = models_rdd.first()[0]
        # logger.debug("length: %d samples:%s ", get_size(sample), sample, exc_info=1)
        size_each_entry_in_models_rdd = get_size(models_rdd.map(lambda x: x[0]).take(5))/5 * 4.0
        logger.info("Avg size of each entry: {}".format(size_each_entry_in_models_rdd))
        batch_size_collect = MAX_RESULT_COLLECT_IN_MB*1024*1024 // (size_each_entry_in_models_rdd*1.5) # 1.5 is the scale-factor

        n_iters = max(1, math.ceil(n_models_rdd*1.0/batch_size_collect))
        logger.info("Collect result of model {} in {} iterations".format(model.name, n_iters))

        csv_writer = None
        file_path = os.path.join(experiment_directory_path, "predicted_{}.csv".format(model.name))
        with open(file_path, "w", newline='') as csv_file:

            for i in range(n_iters):
                _tss = models_rdd\
                        .filter(lambda x: x[1] >= i*batch_size_collect and x[1] < (i+1)*batch_size_collect)\
                        .map(lambda x: x[0]).collect()  

                for _v in _tss:
                    row_to_write = _v[0].to_csv()
                    if csv_writer is None:
                        csv_writer = csv.DictWriter(csv_file, fieldnames=row_to_write.keys(),
                                                    delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writeheader()
                    csv_writer.writerow(row_to_write)
        models_rdd.unpersist()
        models_rdd_wo_idx.unpersist()
    sc.stop()
    logger.info("All Experiments finished. Results are in {}".format(experiment_directory_path))

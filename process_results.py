import argparse
import csv
import logging
import logging.config
import os

import matplotlib.pyplot as plt
from typing import Dict, List

import numpy as np

from core.TimeSeries import TimeSeries


def build_figure(ts: TimeSeries):
    fig = plt.figure()

    plt.plot(ts.observations, label="Observed")
    plt.plot([x.value for x in ts.predictions], label="Predicted")

    plt.legend(loc="best")
    plt.show()


class Metrics:
    def __init__(self):
        pass

    @staticmethod
    def calculate(tss: List[TimeSeries]) -> Dict:
        return {
            "estimation_errors": [ts.estimation_errors() for ts in tss],
            "under_estimation_errors": [ts.under_estimation_errors() for ts in tss],
            "over_estimation_errors": [ts.over_estimation_errors() for ts in tss],
            "estimation_errors_area": [ts.estimation_errors_area() for ts in tss],
            "under_estimation_errors_area": [ts.under_estimation_errors_area() for ts in tss],
            "over_estimation_errors_area": [ts.over_estimation_errors_area() for ts in tss],
            "estimation_percentage_errors": [ts.estimation_percentage_errors() for ts in tss],
            "root_mean_squared_error": [ts.rmse() for ts in tss],
            "with_at_least_one_under_estimation_error": np.sum(
                np.multiply([ts.has_one_under_estimation_error() for ts in tss], 1))
        }

    @staticmethod
    def print(metrics):
        max_metric_width = np.max([len(x) for v in metrics.values() for x in v.keys()])
        max_model_width = np.max([len(x) for x in metrics.keys()])

        for model in sorted(metrics.keys()):
            for metric in sorted(metrics[model].keys()):
                # Let's flatten the array
                # Numpy is giving me problems when casting the matrix to float64 so I cannot use its flatten method
                if isinstance(metrics[model][metric], List) and \
                        all(isinstance(l, List) for l in metrics[model][metric]):
                    _values = [x for _x in metrics[model][metric] for x in _x]
                elif isinstance(metrics[model][metric], List) and \
                        all(isinstance(f, float) for f in metrics[model][metric]):
                    _values = [x for x in metrics[model][metric]]
                else:
                    _values = metrics[model][metric]

                values = np.array(_values, dtype=np.float64)
                logging.info("[{:^{model_width}}][{:^{metric_width}}] "
                             "Mean: {: 10.2f} Median: {: 10.2f} Max: {: 10.2f} Min: {: 10.2f} Size: {}"
                             .format(model, str(metric).replace("_", " ").title(), np.mean(values),
                                     np.median(values), np.max(values), np.min(values), values.size,
                                     model_width=max_model_width, metric_width=max_metric_width))


if __name__ == '__main__':
    logging.config.fileConfig(os.path.join("logging.conf"))
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Experiment results parser.")
    parser.add_argument(action='store', dest='exp_folder', help='Experiment folder path')
    args = parser.parse_args()

    all_tss = {}

    exp_folder = os.path.abspath(args.exp_folder)

    for filename in os.listdir(exp_folder):
        model_name = str(str(filename.split(".")[0]).split("_")[1])
        logging.info("Parsing file for model: {}".format(model_name))

        all_tss[model_name] = []
        with open(os.path.join(exp_folder, filename)) as results_csvfile:
            csv_reader = csv.DictReader(results_csvfile)
            for row in csv_reader:
                all_tss[model_name].append(TimeSeries(**row))

        logging.info("Done.")

    logging.info("Calculate Metrics...")
    all_metrics = {model_name: Metrics.calculate(all_tss[model_name]) for model_name in all_tss}
    logging.info("Done.")

    Metrics.print(all_metrics)

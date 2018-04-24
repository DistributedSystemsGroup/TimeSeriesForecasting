import argparse
import csv
import logging
import logging.config
import os

import matplotlib.pylab as plt
from typing import Dict, List

import numpy as np
import pandas as pd

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
            "mean_absolute_percentage_error": [ts.mape() for ts in tss],
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
    @staticmethod
    def plot_mapeBO(metrics):

        list_norm = []
        list_bo = []
        list_models = []

        for model in sorted(metrics.keys()):
            if model[-2:] == 'BO':
                list_models.append(model[:-2])
                list_bo.append(np.mean(metrics[model]["mean_absolute_percentage_error"]))
            else:
                list_norm.append(np.mean(metrics[model]["mean_absolute_percentage_error"]))

        fig = plt.figure()

        p1, = plt.plot(list_models, list_norm, color="red")
        p2, = plt.plot(list_models, list_bo, color="blue")
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.title("Comparison of different forecasting models", fontweight='bold')
        plt.xlabel("Model", style='italic', fontsize=14)
        plt.ylabel("Average MAPE", style='italic', fontsize=14)
        plt.legend([p1,p2],["Without BO","With BO"])
        fig.autofmt_xdate()
        plt.tight_layout()

        fig.savefig('mape_comparisonBO.png')
        fig.show()

    @staticmethod
    def plot_mape(metrics):

        list_mape = []
        list_models = []

        for model in sorted(metrics.keys()):
            list_models.append(model)
            list_mape.append(np.mean(metrics[model]["mean_absolute_percentage_error"]))

        fig = plt.figure()

        plt.scatter(list_models, list_mape, color="red")
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.title("Comparison of different forecasting models", fontweight='bold')
        plt.xlabel("Model", style='italic', fontsize=14)
        plt.ylabel("Average MAPE", style='italic', fontsize=14)
        fig.autofmt_xdate()
        plt.tight_layout()

        fig.savefig('mape_comparison.png')
        fig.show()

    @staticmethod
    def build_table(metrics):

        list_models = []
        list_mape = []
        list_rmse = []

        for model in sorted(metrics.keys()):
            list_models.append(model)
            list_mape.append(np.mean(metrics[model]["mean_absolute_percentage_error"]))
            list_rmse.append(np.mean(metrics[model]["root_mean_squared_error"]))

        dict_table = {"Models": list_models, "MAPE": list_mape, "RMSE": list_rmse}
        df = pd.DataFrame(dict_table)
        df.set_index("Models", inplace=True, drop=True)
        print(df)

    @staticmethod
    def box_plot(metrics):

        box_list = []

        for model in sorted(metrics.keys()):
            box_list.append(metrics[model]["mean_absolute_percentage_error"])

        fig = plt.figure()

        x_axes = np.arange(1,len(box_list)+1)
        plt.boxplot(box_list, 0, '')

        plt.xticks(x_axes, sorted(metrics.keys()), fontsize=11)
        plt.yticks(fontsize=11)
        plt.title("Comparison of different forecasting models", fontweight='bold')
        plt.xlabel("Model", style='italic', fontsize=14)
        plt.ylabel("MAPE", style='italic', fontsize=14)
        fig.autofmt_xdate()
        plt.tight_layout()

        fig.savefig('mape_boxplots.png')
        fig.show()


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
    Metrics.plot_mape(all_metrics)
    #Metrics.plot_mapeBO(all_metrics)
    Metrics.box_plot(all_metrics)
    Metrics.build_table(all_metrics)

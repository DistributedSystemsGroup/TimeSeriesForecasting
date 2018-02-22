import csv
import os

import matplotlib.pyplot as plt

if __name__ == '__main__':
    results_path = os.path.join("experiment_results", "2018-02-22-15-04-44")

    with open(os.path.join(results_path, "predicted_Arima.csv"), "r") as results_csvfile:
        csv_reader = csv.DictReader(results_csvfile)

        for (i, row) in enumerate(csv_reader):
            if i < 1:
                fig = plt.figure()

                observed_values = [float(v) for v in row["observed"].split(" ")]
                predicted_values = [float(v) for v in row["predicted"].split(" ")]

                # plt.plot(
                #         [np.arange(len(observed_values)), np.arange(len(predicted_values))],
                #          [observed_values, predicted_values],
                #          label=["Observed", "Predicted"])
                plt.plot(observed_values, label="Observed")
                plt.plot(predicted_values, label="Predicted")
                plt.legend(loc="best")
                plt.show()


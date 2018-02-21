import csv
import logging
import os
from collections import OrderedDict

if __name__ == '__main__':
    def m4():
        filename = "Hourly-train.csv"
        output_filename = os.path.splitext(filename)[0] + ".csv"

        filename_info = "M4-info.csv"

        m4_info = {}
        with open(os.path.join(dirty_traces_folder, filename_info)) as csvreadfile_info:
            reader_info = csv.DictReader(csvreadfile_info)

            for row in reader_info:
                if row["M4id"] not in m4_info:
                    m4_info[row["M4id"]] = {"Horizon": row["Horizon"]}
                else:
                    logging.warning("Key already exist. {}".format(row["M4id"]))

        assert len(m4_info) != 0, "We do not have any info regarding the traces in {}".format(filename)

        with open(os.path.join(dirty_traces_folder, filename)) as csvreadfile:
            reader = csv.DictReader(csvreadfile)

            with open(os.path.join(clean_traces_directory_path, output_filename), "w", newline='') as csvwritefile:
                fields_names = ["values", "forecasting_window", "minimum_observations"]
                csv_writer = csv.DictWriter(csvwritefile, fieldnames=fields_names,
                                            delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writeheader()

                for row in reader:
                    # We remove the first column that hold the id of the time series
                    m4id = row.pop("V1")
                    # We strip the first char from the key so that we can sort it numerically
                    _row = {int(k[1:]): v for k, v in row.items() if v != ""}
                    values = OrderedDict(sorted(_row.items())).values()
                    csv_writer.writerow({
                        "values": " ".join(values),
                        "forecasting_window": m4_info[m4id]["Horizon"],
                        "minimum_observations": len(values) - int(m4_info[m4id]["Horizon"])
                    })

    def m3(sheet_name: str):
        filename = "M3C.xls"
        output_filename = os.path.splitext(filename)[0] + ".csv"

        from pandas import read_excel
        df = read_excel(os.path.join(dirty_traces_folder, filename), sheet_name=sheet_name)

        with open(os.path.join(clean_traces_directory_path, output_filename), "w", newline='') as csvwritefile:
            fields_names = ["values", "forecasting_window", "minimum_observations"]
            csv_writer = csv.DictWriter(csvwritefile, fieldnames=fields_names,
                                        delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writeheader()

            for index, row in df.iterrows():
                # The first value of the time series starts from column 6
                values = row.iloc[6:].dropna(how='all').values.tolist()
                csv_writer.writerow({
                    "values": " ".join(str(v) for v in values),
                    "forecasting_window": row["NF"],
                    "minimum_observations": len(values) - int(row["NF"])
                })

    dirty_traces_folder = os.path.join("..", "traces", "dirty_traces")
    clean_traces_folder = os.path.join("..", "traces")

    clean_traces_directory_path = os.path.join(clean_traces_folder)
    if not os.path.exists(clean_traces_directory_path):
        os.makedirs(clean_traces_directory_path)

    m4()
    m3("M3Year")

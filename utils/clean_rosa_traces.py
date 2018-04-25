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

    def m1(sheet_name: str, type_data: str):
        filename = "MC1001.xls"
        output_filename = os.path.splitext(filename)[0] + "_" + type_data[:-2] + ".csv"

        from pandas import read_excel
        df = read_excel(os.path.join(dirty_traces_folder, filename), sheet_name=sheet_name)

        with open(os.path.join(clean_traces_directory_path, output_filename), "w", newline='') as csvwritefile:
            fields_names = ["values", "forecasting_window", "minimum_observations"]
            csv_writer = csv.DictWriter(csvwritefile, fieldnames=fields_names,
                                        delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writeheader()

            for index, row in df.iterrows():
                if row["Type"].strip() == type_data:
                    # The first value of the time series starts from column 7
                    values = row.iloc[7:].dropna(how='all').values.tolist()
                    csv_writer.writerow({
                        "values": " ".join(str(v) for v in values),
                        "forecasting_window": row["NF"],
                        "minimum_observations": len(values) - int(row["NF"])
                    })

    def tcomp(sheet_name: str):
        filename = "MHcomp1.xls"
        output_filename = os.path.splitext(filename)[0] + "_" + sheet_name[:-2] + ".csv"

        from pandas import read_excel
        df = read_excel(os.path.join(dirty_traces_folder, filename), sheet_name=sheet_name)

        NF_dict = {"Yearly": 6, "Quarterly": 8, "Monthly": 12, "Weekly": 26, "Daily": 14, "Hourly": 48}

        with open(os.path.join(clean_traces_directory_path, output_filename), "w", newline='') as csvwritefile:
            fields_names = ["values", "forecasting_window", "minimum_observations"]
            csv_writer = csv.DictWriter(csvwritefile, fieldnames=fields_names,
                                        delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writeheader()

            for index, column in df.iteritems():
                if index != "n":
                    # The first value of the time series starts from row 4
                    values = column.iloc[3:].dropna(how='all').values.tolist()
                    csv_writer.writerow({
                        "values": " ".join(str(v) for v in values),
                        "forecasting_window": NF_dict[sheet_name],
                        "minimum_observations": len(values) - NF_dict[sheet_name]
                    })

    def cif():
        filename = "cif-dataset.csv"
        output_filename = os.path.splitext(filename)[0] + ".csv"

        with open(os.path.join(dirty_traces_folder, filename)) as csvreadfile:
            reader = csv.reader(csvreadfile)

            with open(os.path.join(clean_traces_directory_path, output_filename), "w", newline='') as csvwritefile:
                fields_names = ["values", "forecasting_window", "minimum_observations"]
                csv_writer = csv.DictWriter(csvwritefile, fieldnames=fields_names,
                                            delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writeheader()

                for row in reader:
                    _row = row[0].split(";")
                    values = _row[3:]
                    csv_writer.writerow({
                        "values": " ".join(values),
                        "forecasting_window": int(_row[1]),
                        "minimum_observations": len(values) - int(_row[1])
                    })

    dirty_traces_folder = os.path.join("..", "traces", "dirty_traces")
    clean_traces_folder = os.path.join("..", "traces")

    clean_traces_directory_path = os.path.join(clean_traces_folder)
    if not os.path.exists(clean_traces_directory_path):
        os.makedirs(clean_traces_directory_path)

    m4()
    m3("M3Year")
    m1("MC1001", "YEARLY")
    tcomp("Yearly")
    cif()

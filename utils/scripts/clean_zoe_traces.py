import json
import os
import csv

import pandas as pd


MINIMUM_POINTS = 30
# TARGET_CLASS = "spark"
# CONSTRAINTS = {
#     "spark": ["spark-master", "spark-submit", "spark-worker"]
# }


def load_json(file_paths):
    datas = []
    for file_path in file_paths:
        with open(file_path) as data_file:
            datas.append(json.load(data_file))

    # Once loaded the file, we must group the values correctly by Execution ID and Service Name.
    # We are also classifing the execution based on the type of application executed.
    grouped_data = {}
    for data in datas:
        for res in data[0]["results"]:
            try:
                execution_id = "prod-" + str(res["tags"]["zoe.execution.id"][0])
            except KeyError:
                execution_id = "prod1-" + str(res["tags"]["zoe_execution_id"][0])

            if execution_id not in grouped_data:
                grouped_data[execution_id] = {
                    "class": "",
                    "services": {}
                }

            try:
                service_name = res["tags"]["zoe.service.name"][0]
            except KeyError:
                service_name = res["tags"]["zoe_service_name"][0]

            if service_name not in grouped_data[execution_id]["services"]:
                grouped_data[execution_id]["services"][service_name] = []
            for value in res["values"]:
                grouped_data[execution_id]["services"][service_name].append((value[0], value[1]))

    # Now we are filtering the bad records:
    # (i) that do not have all the components for that application class,
    # (ii) that have less than 20 points (1 point is equal to a minute)
    good_execution_id = []
    for execution_id in grouped_data:
        #     app_class = grouped_data[execution_id]["class"]
        #     if app_class == TARGET_CLASS:
        #         try:
        #             found = [0] * len(CONSTRAINTS[app_class])
        #         except KeyError as e:
        #             print("## WARN: The application class ({}) does not have any components contraints.".format(e))
        #             break

        minimum_points = 999999
        for service in grouped_data[execution_id]["services"]:
            if minimum_points > len(grouped_data[execution_id]["services"][service]):
                minimum_points = len(grouped_data[execution_id]["services"][service])
        #             for (idx, component) in enumerate(CONSTRAINTS[app_class]):
        #                 if component in service:
        #                     found[idx] = 1
        #                     break

        if minimum_points < MINIMUM_POINTS:
            continue

        #         if sum(found) != len(CONSTRAINTS[app_class]):
        #             continue

        good_execution_id.append(execution_id)

    print("## INFO: The good records are {} over {}".format(len(good_execution_id), len(grouped_data)))

    expanded_data = {}

    longest_id = ""
    longest_value = 0
    for execution_id in good_execution_id:
        for service_name in grouped_data[execution_id]["services"]:
            _id = execution_id + "-" + service_name
            if _id not in expanded_data:
                expanded_data[_id] = {
                    "x": [],
                    "y": []
                }
            for (i, tp) in enumerate(sorted(grouped_data[execution_id]["services"][service_name], key=lambda x: x[0])):
                expanded_data[_id]["x"].append(i)
                expanded_data[_id]["y"].append(tp[1] / (1024 ** 2))
            if len(grouped_data[execution_id]["services"][service_name]) > longest_value:
                longest_value = len(grouped_data[execution_id]["services"][service_name])
                longest_id = _id

    print("## INFO: Expanded TS are " + str(len(expanded_data)))
    print("## INFO: ID with highest number of points is " + longest_id + " with " + str(longest_value) + " points.")

    # To easy our life, we now convert the data in Pandas dataframes.
    pds_data = {}
    for execution_id in expanded_data:
        df = pd.DataFrame(expanded_data[execution_id])
        #     df['ds'] = pd.to_datetime(df['ds'],unit='ms')
        #     df.set_index('x', inplace=True)
        pds_data[execution_id] = df

    return pds_data


if __name__ == '__main__':
    dirty_traces_folder = os.path.join("..", "traces", "dirty_traces", "zoe")
    clean_traces_folder = os.path.join("..", "traces")

    clean_traces_directory_path = os.path.join(clean_traces_folder)
    if not os.path.exists(clean_traces_directory_path):
        os.makedirs(clean_traces_directory_path)

    all_dirty_files = [os.path.join(dirty_traces_folder, f)
                       for f in os.listdir(dirty_traces_folder)
                       if os.path.isfile(os.path.join(dirty_traces_folder, f))]

    pds_data = load_json(all_dirty_files)

    with open(os.path.join(clean_traces_directory_path, "zoe.csv"), "w", newline='') as csvwritefile:
        fields_names = ["values"]
        csv_writer = csv.DictWriter(csvwritefile, fieldnames=fields_names,
                                    delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writeheader()

        for ts_name in pds_data:
            csv_writer.writerow({
                "values": " ".join(str(v) for v in pds_data[ts_name]["y"])
            })

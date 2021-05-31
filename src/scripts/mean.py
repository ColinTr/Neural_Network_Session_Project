# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Authors: Troisemaine Colin, Bouchard Guillaume and Inacio Éloïse
"""

import sys
import json
from os import listdir
from os.path import isfile, join
import numpy as np
import datetime

if __name__ == "__main__":
    """
    Computes the mean values of every result files inside the given folder path.
    """

    my_path = sys.argv[1]
    files = [join(my_path, f) for f in listdir(my_path) if isfile(join(my_path, f))]

    print("Files", len(files))

    mean_metric = {
        "train_loss": {
            "val": [],
            "var": []
        },
        "train_acc": {
            "val": [],
            "var": []
        },
        "val_loss": {
            "val": [],
            "var": []
        },
        "val_acc": {
            "val": [],
            "var": []
        }
    }

    for f in range(len(files)):
        with open(files[f]) as json_f:
            d = json.load(json_f)
            if f == 0:
                for key in mean_metric.keys():
                    mean_metric[key]["val"] = [[] for i in d['metric_values']['train_loss']]
                    mean_metric[key]["var"] = [0 for i in d['metric_values']['train_loss']]

            for key in mean_metric.keys():
                for v in range(len(mean_metric[key]["val"])):
                    mean_metric[key]["val"][v].append(d['metric_values'][key][v])

    # Percent to remove at the start and the end of the sorted list of values
    confidence_interval = 0.1

    for key in mean_metric.keys():

        # We calculate the nb of value to delete
        nb_to_delete = round(len(mean_metric[key]["val"]) * confidence_interval)

        for v in range(len(mean_metric[key]["val"])):

            # We sort the list of values and remove the n lowest and n highest values.
            values = np.sort(mean_metric[key]["val"][v])[nb_to_delete:-nb_to_delete]

            # Calculate variance
            mean_metric[key]["var"][v] = np.var(values)

            # Calculate mean
            mean_metric[key]["val"][v] = np.mean(values)

    data = d
    mean_metric['instance_queries'] = data['metric_values']['instance_queries']
    data['metric_values'] = mean_metric

    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = "{}-{}-{}-{}.json".format(data["strategy"], data["dataset_name"],  data["model"], date)
    output_path = "../results/mean/{}".format(filename)

    with open(output_path, "w") as f:
        json.dump(data, f)

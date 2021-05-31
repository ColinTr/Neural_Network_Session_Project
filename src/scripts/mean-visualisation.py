# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Authors: Troisemaine Colin, Bouchard Guillaume and Inacio Éloïse
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import sys


if __name__ == "__main__":
    """
    This script is used to plot a superposition of any number of result files
    with the following metrics : val acc, val loss, train acc and train loss.
    """

    if len(sys.argv) <= 1:
        print("This script is used to plot a superposition of any number of result files.")
        print("Usage : python visualisation.py [results file 1] [results file 2] ...")

    results_list = []
    for arg in sys.argv[1:]:
        with open(arg, 'r') as json_file:
            results_list.append(json.load(json_file))

    labelled_examples = []
    for val in results_list[0]['metric_values']['instance_queries']:
        if len(labelled_examples) > 0:
            labelled_examples.append(labelled_examples[-1] + val)
        else:
            labelled_examples.append(results_list[0]['seed_size'] + val)

    fig, ax = plt.subplots()
    title = "{} - {}".format(results_list[0]["dataset_name"], results_list[0]["model"])
    fig.suptitle(title)

    ax.set_xlabel('Labelled examples')
    ax.set_ylabel('Validation accuracy')

    colors = ['b', 'r', 'g', 'c', 'm', 'y']

    for results, color in zip(results_list, colors[:len(results_list)]):
        acc_val = np.array(results['metric_values']['val_acc']['val'])
        acc_var = np.array(results['metric_values']['val_acc']['var'])
        ax.plot(labelled_examples, acc_val, '-o', label=results['strategy'])
        plt.fill_between(labelled_examples, acc_val - acc_var, acc_val + acc_var, color=color, alpha=0.2)

    ax.legend()

    fig.savefig('full_results_fig.png')
    plt.show()

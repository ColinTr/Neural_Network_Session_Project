# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Authors: Troisemaine Colin, Bouchard Guillaume and Inacio Éloïse
"""

import matplotlib.pyplot as plt
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

    f = plt.figure(figsize=(10, 10))
    title = "{} - {}".format(results_list[0]["dataset_name"], results_list[0]["model"])
    f.suptitle(title)

    ax1 = f.add_subplot(221)
    ax1.set_xlabel('Labelled examples')
    ax1.set_ylabel('Validation accuracy')

    ax2 = f.add_subplot(222)
    ax2.set_xlabel('Labelled examples')
    ax2.set_ylabel('Training accuracy')

    ax3 = f.add_subplot(223)
    ax3.set_xlabel('Labelled examples')
    ax3.set_ylabel('Validation loss')

    ax4 = f.add_subplot(224)
    ax4.set_xlabel('Labelled examples')
    ax4.set_ylabel('Training loss')

    for results in results_list:
        ax1.plot(labelled_examples, results['metric_values']['val_acc'], '-o', label=results['strategy'])
        ax2.plot(labelled_examples, results['metric_values']['train_acc'], '-o', label=results['strategy'])
        ax3.plot(labelled_examples, results['metric_values']['val_loss'], '-o', label=results['strategy'])
        ax4.plot(labelled_examples, results['metric_values']['train_loss'], '-o', label=results['strategy'])

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()

    f.savefig('full_results_fig.png')
    plt.show()

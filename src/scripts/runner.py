# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Authors: Troisemaine Colin, Bouchard Guillaume and Inacio Éloïse
"""

import os

if __name__ == "__main__":
    """
    Allows to launch sequentially any number of train.py scripts to generate results.
    """

    seed_size = "--seed_size={}".format(100)
    query_size = "--query_size={}".format(100)
    stopping_criterion_value = "--stopping_criterion_value={}".format(900)
    num_epochs = "--num_epochs={}".format(20)
    validation = "--validation={}".format(0.1)
    batch_size = "--batch_size={}".format(10)

    basic_args = "{} {} {} {} {} {}".format(seed_size, query_size, stopping_criterion_value, num_epochs,
                                            validation, batch_size)

    cmd = [
        "python train.py {} --model=vggnet --dataset=mnist --strategy=diverse_mini_batch".format(basic_args),
        "python train.py {} --model=vggnet --dataset=mnist --strategy=random_sampling".format(basic_args)
    ]

    repeat = 10
    for c in cmd:
        print("Launching on {} repetition :\n{}".format(repeat, c))
        for i in range(repeat):
            print("Repetition {}/{}".format(i + 1, repeat))
            os.system(c)

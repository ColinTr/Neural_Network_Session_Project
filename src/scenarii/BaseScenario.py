# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Authors: Troisemaine Colin, Bouchard Guillaume and Inacio Éloïse
"""


class BaseScenario:
    """
    Base class that represents a scenario in which we generally place ourselves when we do active learning.

    We can mainly cite three scenarios :
        - 'Membership Query Synthesis'
        - 'Stream-Based Selective Sampling'
        - 'Pool-Based Sampling'
    """

    def __init__(self, strategy):
        """
        :param strategy: the strategy that will be used with our scenario
        """
        self.strategy = strategy
        self.strategies = {}

    def query_strategy(self, prediction, **kwargs):
        """
        Use the model to get predictions on the dataset and then apply the strategy on these predictions.
        :param prediction: The prediction in which the indexes will be selected
        """
        try:
            strat = self.strategies[self.strategy]
            return strat(prediction, **kwargs)

        except KeyError as err:
            raise Exception("There is no strategy named {} for this scenario.".format(self.strategy)) from err

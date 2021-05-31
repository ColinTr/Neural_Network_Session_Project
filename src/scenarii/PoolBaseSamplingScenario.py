# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Authors: Troisemaine Colin, Bouchard Guillaume and Inacio Éloïse
"""

from scenarii.BaseScenario import BaseScenario
from torch.distributions import Categorical
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import random
import torch


class PoolBaseSamplingScenario(BaseScenario):
    """
    Scenario that assumes that we have a pool of unlabelled data waiting to be labelled. The model will make the
    prediction on a batch, and we will select a k (query_size) number of these predictions to send to the oracle.
    This scenario return a list of indexes that represents the indexes of the selected prediction in the given tensor
    of predictions.
    """

    def __init__(self, strategy="least_confidence", query_size=8):
        """
        :param strategy: the query strategy that will be used in the scenario
        :param query_size: specific variable for this scenario, decide the size the length of the list that
        we will return
        """
        super().__init__(strategy)

        self.strategies = {
            "least_confidence": self._least_confidence_strategy,
            "entropy_sampling": self._entropy_sampling_strategy,
            "margin_sampling": self._margin_sampling_strategy,
            "random_sampling": self._random_sampling_strategy,
            "diverse_mini_batch": self._diverse_mini_batch_strategy
        }

        self.query_size = query_size

    def _least_confidence_strategy(self, prediction, **kwargs):
        """
        A strategy where we select the k (query_size) predictions which have the least confidence in their most likely
        label prediction.
        :param prediction: A prediction (a softmax must be applied before calling this method)
        :return: The indexes that were selected by the strategy
        """
        batch_size, _ = prediction.shape
        if batch_size < self.query_size:
            raise ValueError("The query size is bigger than the batch size")

        # We get the max value of each prediction
        out, _ = torch.max(prediction, 1)

        # We get the indexes of the k (query_size) minimum value
        _, inds = torch.topk(out, self.query_size, largest=False)

        return inds

    def _entropy_sampling_strategy(self, prediction, **kwargs):
        """
        A strategy where we select the k (query_size) predictions which have the highest entropy score.
        :param prediction: A prediction (a softmax must be applied before calling this method)
        :return: The indexes that were selected by the strategy
        """
        batch_size, _ = prediction.shape
        if batch_size < self.query_size:
            raise ValueError("The query size is bigger than the batch size")

        # We calculate the entropy for each prediction in our tensor
        ent = Categorical(prediction).entropy()

        # We get the indexes of the k (query_size) maximum value
        _, inds = torch.topk(ent, self.query_size, largest=True)

        return inds

    def _margin_sampling_strategy(self, prediction, custom_query_size=None, **kwargs):
        """
        A strategy where we select the k (query_size) predictions which have the lowest difference between their
        two most likely label prediction.
        :param prediction: A prediction (a softmax must be applied before calling this method)
        :return: The indexes that were selected by the strategy
        """
        if custom_query_size is None:
            qs = self.query_size
        else:
            qs = custom_query_size

        batch_size, _ = prediction.shape
        if batch_size < qs:
            raise ValueError("The query size is bigger than the batch size")

        # We get the max value of each prediction
        out, _ = torch.topk(prediction, 2)
        out = torch.flatten(torch.diff(out))

        # We get the indexes of the k (query_size) minimum value
        _, inds = torch.topk(out, qs, largest=False)

        return inds

    def _random_sampling_strategy(self, prediction, **kwargs):
        """
        A strategy where we select the k (query_size) predictions at random.
        :param prediction: A prediction
        :return: The indexes that were selected by the strategy
        """
        batch_size, _ = prediction.shape
        if batch_size < self.query_size:
            raise ValueError("The query size is bigger than the batch size")

        indexes = random.sample(range(batch_size), self.query_size)
        indexes = torch.LongTensor(indexes)

        return indexes

    def _diverse_mini_batch_strategy(self, prediction, data_manager, k_clusters, beta=50, **kwargs):
        """

        :param prediction:
        :return:
        """
        # We start by applying a smallest margin query sampler :

        # If we don't have enough data, we will only consider the data we have in our margin sampling
        custom_query_size = self.query_size * beta
        points_number, _ = prediction.shape

        if custom_query_size > points_number:
            custom_query_size = points_number

        pre_selected_samples = self._margin_sampling_strategy(prediction=prediction,
                                                              custom_query_size=custom_query_size)

        for i, data in enumerate(data_manager.get_train_data_by_index(pre_selected_samples), 0):
            # transfer tensors to selected device
            train_inputs = data[0].to('cpu')
            train_inputs = train_inputs.numpy()
            X = np.array([x.flatten() for x in train_inputs])
            variance = 0.98
            pca = PCA(variance)
            pca.fit(X)
            X = pca.transform(X)

            k_means = KMeans(init='k-means++', n_clusters=k_clusters)
            k_means.fit(X)

            distances = []
            for index in range(len(k_means.labels_)):
                label = k_means.labels_[index]
                centroid = k_means.cluster_centers_[label]
                distances.append(np.linalg.norm(X[index] - centroid))

            # We finish by taking the self.query_size smallest distances :
            ksorted_indexes = np.argpartition(distances, self.query_size)[:self.query_size]

            return pre_selected_samples[ksorted_indexes]

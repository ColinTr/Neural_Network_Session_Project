# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Authors: Troisemaine Colin, Bouchard Guillaume and Inacio Éloïse
Inspired from :
    University of Sherbrooke
    Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
"""

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import torch


class DataManager(object):
    """
    class that yields dataloaders for train, test, and validation data
    Inspired from :
        University of Sherbrooke
        Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
    """

    def __init__(self,
                 train_dataset: Dataset,
                 test_dataset: Dataset,
                 seed_size: int = 100,
                 batch_size: int = 1,
                 validation_proportion: float = 0.1,
                 seed: int = 0):
        """
        :param train_dataset: the train Dataset
        :param test_dataset: The test Dataset
        :param seed_size: Initial size of the seed dataset
        :param batch_size: Batch size used in the seed, train, val and test datasets
        :param validation_proportion: float, proportion of the train dataset used for the validation set
        :param seed: int, random seed for splitting train and validation set
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.seed_size = seed_size
        self.batch_size = batch_size
        self.validation_proportion = validation_proportion
        self.seed = seed

        self.seed_indexes, self.train_indexes, self.val_indexes = self.train_validation_split(seed_size,
                                                                                              len(train_dataset),
                                                                                              validation_proportion,
                                                                                              seed)

        self.seed_loader = None
        self.train_loader = None
        self.predict_train_loader = None
        self.validation_loader = None
        self.test_loader = None

        self.generate_data_loaders()

    def generate_data_loaders(self):
        """
        Uses the indexes of the class to generate the DataLoader objects.
        """
        torch.manual_seed(self.seed)

        seed_sampler = SubsetRandomSampler(self.seed_indexes)
        train_sampler = SubsetRandomSampler(self.train_indexes)
        val_sampler = SubsetRandomSampler(self.val_indexes)

        self.seed_loader = DataLoader(self.train_dataset, self.batch_size, sampler=seed_sampler)
        self.train_loader = DataLoader(self.train_dataset, self.batch_size, sampler=train_sampler)
        self.predict_train_loader = DataLoader(self.train_dataset, 500, sampler=train_sampler)
        self.validation_loader = DataLoader(self.train_dataset, self.batch_size, sampler=val_sampler)
        self.test_loader = DataLoader(self.test_dataset, self.batch_size, shuffle=True)

    @staticmethod
    def train_validation_split(seed_size, num_samples, validation_ratio, seed=0):
        """
        Returns 3 torch Samplers, one for seed, one for training, the other for validation.
        All samplers are based on the training dataset (see __init__).
        :param seed_size: size of the seed dataset
        :param num_samples: number of samples to split between train and val set
        :param validation_ratio: proportion of the train dataset used for validation set
        :param seed: random seed for splitting train and validation set
        :return: train and validation samplers
        """
        torch.manual_seed(seed)
        num_val = int(num_samples * validation_ratio)
        shuffled_idx = torch.randperm(num_samples).long()

        tmp_train_idx = shuffled_idx[num_val:]

        seed_indexes = tmp_train_idx[:seed_size]
        train_indexes = tmp_train_idx[seed_size:]
        val_indexes = shuffled_idx[:num_val]

        return seed_indexes, train_indexes, val_indexes

    def add_indexes_to_seed_sampler(self, train_indexes_to_add):
        """
        Takes the given indexes from the train dataset and puts them into the seed dataset.
        So the seed dataset grows after every call of this method.
        :param train_indexes_to_add:
        """
        # print("previous seed_indexes size :", len(self.seed_indexes))
        # print("previous train_indexes size :", len(self.train_indexes))
        # print("len(train_indexes_to_add)=", len(train_indexes_to_add))
        self.seed_indexes = torch.cat((self.seed_indexes, self.train_indexes[train_indexes_to_add]))
        self.train_indexes[train_indexes_to_add] = -1
        self.train_indexes = self.train_indexes[self.train_indexes[:] >= 0]
        # print("new seed_indexes size :", len(self.seed_indexes))
        # print("new train_indexes size :", len(self.train_indexes))

        # Regenerate samplers
        self.generate_data_loaders()

    def get_seed_set(self):
        """
        :return: The seed dataset loader
        """
        return self.seed_loader

    def get_train_set(self):
        """
        :return: The train dataset loader
        """
        return self.train_loader

    def get_predict_train_set(self):
        """
        :return: train dataset loader with a larger batch size (used for prediction)
        """
        return self.predict_train_loader

    def get_validation_set(self):
        """
        :return: The validation dataset loader
        """
        return self.validation_loader

    def get_test_set(self):
        """
        :return: The test dataset loader
        """
        return self.test_loader

    def get_train_data_by_index(self, indexes):
        """
        Returns a DataLoader which will give one single batch of all of the training data based on the indexes
        passed in parameters
        :param indexes: the indexes of the list self.train_indexes to select
        :return: The DataLoader object of the wanted data points
        """
        sampler = SubsetRandomSampler(self.train_indexes[indexes])
        return DataLoader(self.train_dataset, len(indexes), sampler=sampler)

    def get_random_sample_from_test_set(self):
        """
        :return: A random sample from the test dataset
        """
        index = np.random.randint(0, len(self.test_dataset))
        return self.test_dataset[index]

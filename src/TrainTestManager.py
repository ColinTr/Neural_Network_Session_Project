# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Authors: Troisemaine Colin, Bouchard Guillaume and Inacio Éloïse
Inspired from :
    University of Sherbrooke
    Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
"""

from torch.utils.data import Dataset
from DataManager import DataManager
from typing import Callable, Type
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import warnings
import datetime
import torch
import json


class TrainTestManager(object):
    """
    Class used the train and test the given model in the parameters
    """

    def __init__(self,
                 dnn_model,
                 train_set: Dataset,
                 test_set: Dataset,
                 num_epochs: int,
                 scenario,
                 loss_fn: torch.nn.Module,
                 optimizer_factory: Callable[[torch.nn.Module], torch.optim.Optimizer],
                 stopping_criterion: str = "query_number",
                 stopping_criterion_value: int = 100,
                 validation_proportion: int = 0.1,
                 batch_size: int = 1,
                 query_size=10,
                 seed_size: int = 100,
                 use_cuda: bool = True,
                 dataset_name="undefined"):
        """
        :param dnn_model: model to train
        :param train_set: dataset to train the model
        :param test_set: dataset to test the model
        :param stopping_criterion: criterion used to stop active learning
        :param num_epochs: number of epochs per training iteration
        :param scenario: the active learning scenario to use
        :param stopping_criterion_value: value used with stopping_criterion if needed (ex : max number of queries)
        :param validation_proportion: proportion of the train dataset to use for validation
        :param batch_size: batch size used in training
        :param seed_size: initial size of the training dataset for active learning
        :param use_cuda: use the GPU or the CPU for computation (True=GPU, False=CPU)
        """
        self.data_manager = DataManager(train_set, test_set, seed_size=seed_size, batch_size=batch_size,
                                        validation_proportion=validation_proportion)
        self.num_epochs = num_epochs
        self.scenario = scenario

        if stopping_criterion not in ["query_number", "iteration_number", "performance_stagnation"]:
            self.stopping_criterion = "query_number"
            self.stopping_criterion_value = 100
            warnings.warn("Unknown stopping_criterion, using default value \"query_number\" instead.")
        else:
            self.stopping_criterion = stopping_criterion
            self.stopping_criterion_value = stopping_criterion_value

        self.batch_size = batch_size
        self.query_size = query_size
        self.seed_size = seed_size
        self.use_cuda = use_cuda
        self.dataset_name = dataset_name

        device_name = 'cuda:0' if self.use_cuda else 'cpu'
        if self.use_cuda and not torch.cuda.is_available():
            warnings.warn("CUDA is not available. Suppress this warning by passing "
                          "use_cuda=False to {}()."
                          .format(self.__class__.__name__), RuntimeWarning)
            device_name = 'cpu'

        self.device = torch.device(device_name)
        self.dnn_model = dnn_model
        self.dnn_model = self.dnn_model.to(self.device)
        self.optimizer = optimizer_factory(self.dnn_model)
        self.loss_fn = loss_fn
        self.metric_values = {}  # Dict of the metrics

    def active_learning_training(self, num_epochs):
        """
        Main method which hosts all of the logic of the active learning.
        It also saves the results in json in the file ./results/... so they can be plotted later
        :param num_epochs: number of epochs per training iteration
        """
        # We first need to save the model so we can restart from the same weights later
        self.dnn_model.save("temp_save.pt")

        # We start by training on a seed sized dataset
        seed_loader = self.data_manager.get_seed_set()
        print("Size of seed dataset : ", len(self.data_manager.seed_indexes))

        last_train_loss, last_train_acc = self.train(seed_loader, num_epochs)

        val_loss, val_acc = self.evaluate_on_validation_set()

        # Initialize metrics container
        self.metric_values['train_loss'] = [last_train_loss]
        self.metric_values['train_acc'] = [last_train_acc]
        self.metric_values['val_loss'] = [val_loss]
        self.metric_values['val_acc'] = [val_acc]
        self.metric_values['instance_queries'] = [0]

        # Then start the active learning until we reach the stopping criterion
        if self.stopping_criterion == "query_number":
            while sum(self.metric_values['instance_queries']) < self.stopping_criterion_value:
                # First we start by querying which indices of the unlabelled dataset we want to add in our training
                # dataset
                prediction = self.predict_train_dataset()

                # Apply a softmax function before gicing it to the strategy
                prediction = torch.softmax(prediction, dim=1)

                # Ask the indexes to keep to the query scenario's query strategy
                if self.scenario.strategy == "diverse_mini_batch":
                    dataset_indexes_to_add = self.scenario.query_strategy(prediction,
                                                                          data_manager=self.data_manager,
                                                                          k_clusters=len(
                                                                              self.data_manager.train_dataset.classes))
                else:
                    dataset_indexes_to_add = self.scenario.query_strategy(prediction)

                # We then add them to our training dataset
                self.data_manager.add_indexes_to_seed_sampler(dataset_indexes_to_add)
                print("Size of training dataset : ", len(self.data_manager.seed_indexes))

                seed_loader = self.data_manager.get_seed_set()

                # We need to reset the model's weights before training it
                self.dnn_model.load_weights("temp_save.pt")

                last_train_loss, last_train_acc = self.train(seed_loader, num_epochs)

                val_loss, val_acc = self.evaluate_on_validation_set()

                # Update metrics container
                self.metric_values['train_loss'].append(last_train_loss)
                self.metric_values['train_acc'].append(last_train_acc)
                self.metric_values['val_loss'].append(val_loss)
                self.metric_values['val_acc'].append(val_acc)
                self.metric_values['instance_queries'].append(len(dataset_indexes_to_add))

        with open('../results/' +
                  self.scenario.strategy + "-" +
                  self.dataset_name + "-" +
                  str(self.dnn_model.__class__.__name__) + "-"
                  + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                  + '_results.json',
                  'w') as outfile:
            data_to_dump = {
                'num_epochs': self.num_epochs,
                'scenario': str(self.scenario.__class__.__name__),
                'strategy': self.scenario.strategy,
                'batch_size': self.batch_size,
                'query_size': self.query_size,
                'seed_size': self.seed_size,
                'model': str(self.dnn_model.__class__.__name__),
                'stopping_criterion': self.stopping_criterion,
                'stopping_criterion_value': self.stopping_criterion_value,
                'dataset_name': self.dataset_name,
                'metric_values': self.metric_values
            }

            json.dump(data_to_dump, outfile)

    def train(self, train_loader, num_epochs):
        """
        Train the model for num_epochs times
        :param num_epochs: number of times to train the model
        :param train_loader: the train dataset to use in training
        """
        # switch to train mode
        self.dnn_model.train()

        last_train_loss = 0
        last_train_acc = 0

        # train num_epochs times
        print("Training model :")
        with tqdm(range(num_epochs * len(train_loader))) as t:
            for epoch in range(num_epochs):
                # print("Epoch: {} of {}".format(epoch + 1, num_epochs))
                train_loss = 0.0

                train_losses = []
                train_accuracies = []
                for i, data in enumerate(train_loader, 0):
                    # transfer tensors to selected device
                    train_inputs, train_labels = data[0].to(self.device), data[1].to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward pass
                    train_outputs = self.dnn_model(train_inputs)
                    # computes loss using loss function loss_fn
                    loss = self.loss_fn(train_outputs, train_labels)

                    # Use autograd to compute the backward pass.
                    loss.backward()

                    # updates the weights using gradient descent
                    """
                    Way it could be done manually

                    with torch.no_grad():
                        for param in self.model.parameters():
                            param -= learning_rate * param.grad
                    """
                    self.optimizer.step()

                    # Save losses for plotting purposes
                    train_losses.append(loss.item())
                    train_accuracies.append(self.accuracy(train_outputs, train_labels))

                    # print metrics along progress bar
                    train_loss += loss.item()
                    t.set_postfix(loss='{:05.3f}'.format(train_loss / (i + 1)))
                    t.update()

                last_train_loss = np.mean(train_losses)
                last_train_acc = np.mean(train_accuracies)

        return last_train_loss, last_train_acc

    def evaluate_on_validation_set(self):
        """
        function that evaluates the model on the validation set every epoch
        :return: val_loss, val_acc
        """
        # switch to eval mode so that layers like batchnorm's layers nor dropout's layers
        # works in eval mode instead of training mode
        self.dnn_model.eval()

        # Get validation data
        val_loader = self.data_manager.get_validation_set()
        validation_loss = 0.0
        validation_losses = []
        validation_accuracies = []

        with torch.no_grad():
            for j, val_data in enumerate(val_loader, 0):
                # transfer tensors to the selected device
                val_inputs, val_labels = val_data[0].to(self.device), val_data[1].to(self.device)

                # forward pass
                val_outputs = self.dnn_model(val_inputs)

                # compute loss function
                loss = self.loss_fn(val_outputs, val_labels)
                validation_losses.append(loss.item())
                validation_accuracies.append(self.accuracy(val_outputs, val_labels))
                validation_loss += loss.item()

        val_loss = np.mean(validation_losses)
        val_acc = np.mean(validation_accuracies)

        # displays metrics
        print('Validation loss %.3f' % (validation_loss / len(val_loader)))

        # switch back to train mode
        self.dnn_model.train()

        return val_loss, val_acc

    def predict_train_dataset(self):
        """
        Note that here, seed_dataset is our real training dataset, so here we predict on "train_dataset" which is kind
        of a validation dataset.
        :return: output of the prediction
        """
        self.dnn_model.eval()

        full_output = []

        # Get validation data
        train_loader = self.data_manager.get_predict_train_set()
        with torch.no_grad():
            print("Predicting train dataset :")
            with tqdm(range(len(train_loader))) as t:
                for j, train_data in enumerate(train_loader, 0):
                    # transfer tensors to the selected device
                    train_inputs, train_labels = train_data[0].to(self.device), train_data[1].to(self.device)

                    # forward pass
                    train_outputs = self.dnn_model(train_inputs)

                    full_output.append(train_outputs.to('cpu'))

                    t.update()

        full_output = torch.cat(full_output, dim=0)

        return full_output

    def evaluate_on_test_set(self):
        """
        Evaluate the model on the test set.
        :returns: Accuracy of the model on the test set
        """
        test_loader = self.data_manager.get_test_set()
        accuracies = 0
        with torch.no_grad():
            for data in test_loader:
                test_inputs, test_labels = data[0].to(self.device), data[1].to(self.device)
                test_outputs = self.dnn_model(test_inputs)
                accuracies += self.accuracy(test_outputs, test_labels)
        print("Accuracy (or Dice for UNet) on the test set: {:05.3f} %".format(100 * accuracies / len(test_loader)))

    def accuracy(self, outputs, labels):
        """
        Computes the accuracy of the model
        :param outputs: outputs predicted by the model
        :param labels: real outputs of the data
        :return: Accuracy of the model
        """
        predicted = outputs.argmax(dim=1)
        correct = (predicted == labels).sum().item()
        return correct / labels.size(0)

    def plot_metrics(self):
        """
        Function that plots train and validation losses and accuracies after training phase
        """
        labelled_examples = []
        for val in self.metric_values['instance_queries']:
            if len(labelled_examples) > 0:
                labelled_examples.append(labelled_examples[-1] + val)
            else:
                labelled_examples.append(self.seed_size + val)

        f = plt.figure(figsize=(10, 5))
        ax1 = f.add_subplot(121)
        ax2 = f.add_subplot(122)

        # loss plot
        ax1.plot(labelled_examples, self.metric_values['train_loss'], '-o', label='Training loss')
        ax1.plot(labelled_examples, self.metric_values['val_loss'], '-o', label='Validation loss')
        ax1.set_title('Training and validation loss')
        ax1.set_xlabel('Labelled examples')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # accuracy plot
        ax2.plot(labelled_examples, self.metric_values['train_acc'], '-o', label='Training accuracy')
        ax2.plot(labelled_examples, self.metric_values['val_acc'], '-o', label='Validation accuracy')
        ax2.set_title('Training and validation accuracy')
        ax2.set_xlabel('labelled examples')
        ax2.set_ylabel('accuracy')
        ax2.legend()
        f.savefig('fig1.png')
        plt.show()


def optimizer_setup(optimizer_class: Type[torch.optim.Optimizer], **hyperparameters) -> \
        Callable[[torch.nn.Module], torch.optim.Optimizer]:
    """
    Creates a factory method that can instanciate optimizer_class with the given
    hyperparameters.

    Why this? torch.optim.Optimizer takes the model's parameters as an argument.
    Thus we cannot pass an Optimizer to the CNNBase constructor.

    Args:
        optimizer_class: optimizer used to train the model
        **hyperparameters: hyperparameters for the model
        Returns:
            function to setup the optimizer
    """

    def f(model):
        return optimizer_class(model.parameters(), **hyperparameters)

    return f

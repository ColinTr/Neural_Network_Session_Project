# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Authors: Troisemaine Colin, Bouchard Guillaume and Inacio Éloïse
Inspired from :
    University of Sherbrooke
    Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
"""

from scenarii.PoolBaseSamplingScenario import PoolBaseSamplingScenario
from TrainTestManager import TrainTestManager, optimizer_setup
from torchvision.transforms import transforms
from models.ResNeXt import ResNeXt
from torchvision import datasets
from models.VGGNet import VggNet
from models.LeNet import LeNet
import torch.optim as optim
import torch.nn as nn
import argparse
import torch


def argument_parser():
    """
    A parser to allow user to easily experiment different models along with datasets and different parameters
    Note that it always uses active learning.
    """

    parser = argparse.ArgumentParser(usage='\n python3 train.py [model] [dataset] [scenario] [strategy] ['
                                           'hyper_parameters]',
                                     description="This program allows to train different models of classification on"
                                                 " different datasets while using active learning.")

    parser.add_argument('--model', type=str, default="vggnet", choices=["vggnet", "resnext", "lenet"])
    parser.add_argument('--dataset', type=str, default="cifar10", choices=["cifar10", "mnist"])
    parser.add_argument('--scenario', type=str, default="poolbasedsampling", choices=["poolbasedsampling"])
    parser.add_argument('--strategy', type=str, default="least_confidence",
                        choices=["least_confidence", "entropy_sampling", "margin_sampling", "random_sampling",
                                 "diverse_mini_batch"])
    parser.add_argument('--stopping_criterion', type=str, default="query_number",
                        choices=["query_number", "iteration_number", "performance_stagnation"])
    parser.add_argument('--stopping_criterion_value', type=int, default=100,
                        help='The value of the stopping criterion (if needed)')
    parser.add_argument('--seed_size', type=int, default=100,
                        help='The size of the initial training set')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='The size of the training batch')
    parser.add_argument('--query_size', type=int, default=10,
                        help='The number of data to add to the training set after each query')
    parser.add_argument('--optimizer', type=str, default="Adam", choices=["Adam", "SGD"],
                        help="The optimizer to use for training the model")
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='The number of epochs per training phase')
    parser.add_argument('--validation', type=float, default=0.1,
                        help='Percentage of training data to use for validation')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')

    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()

    if args.dataset == 'cifar10':
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = datasets.CIFAR10(root='../data', train=True, download=True, transform=base_transform)
        test_set = datasets.CIFAR10(root='../data', train=False, download=True, transform=base_transform)
        num_channels = 3
    elif args.dataset == 'mnist':
        base_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)])
        train_set = datasets.MNIST(root='../data', train=True, download=True, transform=base_transform)
        test_set = datasets.MNIST(root='../data', train=False, download=True, transform=base_transform)
        num_channels = 1

    if args.optimizer == 'SGD':
        optimizer_factory = optimizer_setup(torch.optim.SGD, lr=args.lr, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer_factory = optimizer_setup(optim.Adam, lr=args.lr)

    if args.model == 'vggnet':
        model = VggNet(num_classes=len(train_set.classes), in_channels=num_channels)
    elif args.model == 'resnext':
        model = ResNeXt(num_classes=len(train_set.classes), in_channels=num_channels)
    elif args.model == 'lenet':
        model = LeNet(num_classes=len(train_set.classes), in_channels=num_channels)

    if args.scenario == 'poolbasedsampling':
        scenario = PoolBaseSamplingScenario(strategy=args.strategy, query_size=args.query_size)

    model_trainer = TrainTestManager(dnn_model=model,
                                     train_set=train_set,
                                     test_set=test_set,
                                     num_epochs=args.num_epochs,
                                     scenario=scenario,
                                     loss_fn=nn.CrossEntropyLoss(),
                                     optimizer_factory=optimizer_factory,
                                     stopping_criterion=args.stopping_criterion,
                                     stopping_criterion_value=args.stopping_criterion_value,
                                     validation_proportion=args.validation,
                                     batch_size=args.batch_size,
                                     query_size=args.query_size,
                                     seed_size=args.seed_size,
                                     use_cuda=True,
                                     dataset_name=args.dataset)

    print("Training {} on {} for {} epochs".format(model.__class__.__name__, args.dataset, args.num_epochs))
    model_trainer.active_learning_training(args.num_epochs)
    model_trainer.evaluate_on_test_set()

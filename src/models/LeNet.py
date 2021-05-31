# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Authors: Troisemaine Colin, Bouchard Guillaume and Inacio Éloïse
Inspired from :
    University of Sherbrooke
    Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
"""

import torch
import torch.nn as nn
from models.DNNBaseModel import CNNBaseModel


class LeNet(CNNBaseModel):

    def __init__(self, num_classes=10, init_weights=True, in_channels=3):
        """
        Builds LeNet5 model.

        Args:
            num_classes(int): number of classes.

            init_weights(bool): when true uses _initialize_weights function to initialize
            network's weights.

            in_channels(int): number of channels of input
        """
        super(LeNet, self).__init__(num_classes, init_weights)

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Tensor
        """
        output = self.feature(x)
        output = torch.flatten(output, 1)
        output = self.classifier(output)
        return output

# Neural Network Session Project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Directory Structure

To use train.py and visualisation.py, you need to be in the ./src folder. 

    .
    ├── .gitignore
    ├── AUTHORS.md                          <- The names of the project's authors
    ├── README.md                           <- This file
    ├── requirements.txt                    <- The required packages
    ├── results                             <- The generated results
    └── src                                 <- The source code
        ├── models                          <- Base class, base blocks and implementation of the models
        │   ├── DNNBaseModel.py             <- The base class of which our model's inherit
        │   ├── DNNBlocks.py                <- The base blocks which make up our models
        │   ├── LeNet.py                    <- The LeNet neural network
        │   ├── ResNeXt.py                  <- WIP : The ResNeXt neural network
        │   └── VGGNet.py                   <- The VGGNet neural network
        ├── scenarii                        <- Main scenario and strategies
        │   ├── BaseScenario.py             <- The base class of which our scenarios inherit
        │   └── PoolBaseSamplingScenario.py <- The implementation of the pool sampling scenario and its strategies
        ├── scripts                         <- The scripts detailed below
        │   ├── launch_calcul_canada.sh     <- Used to launch runner.py on calcul canada
        │   ├── mean.py                     <- Used to compute the mean values of all results in a folder
        │   ├── mean-visualisation.py       <- Plots the computed mean values
        │   ├── runner.py                   <- Launches sequentially train.py instances
        │   └── visualisation.py            <- Plots the various metrics of a single result file
        ├── DataManager.py                  <- The manager of the DataLoader objects
        ├── train.py                        <- The main entry point of our project
        └── TrainTestManager.py             <- The class which hosts the active learning's logic
   
## Requirements

This project is meant to be used with python 3.7.

   
## Usage

### Training :

The main entry point script is **train.py**. It launches one full active learning loop.
To use it, type :

> cd src (important !)
> 
> python3 scripts/train.py model dataset [parameters]

The parameters are :\
--model : The model to use (values : vggnet / lenet) (default = vggnet).\
--dataset : The dataset to use (values : CIFAR-10 / mnist) (default = cifar10).\
--strategy : The data selection criterion (values : least_confidence / entropy_sampling / margin_sampling / random_sampling) (default = least_confidence).\
--stopping_criterion_value : The quantity of data to add to the seed dataset (default = 100).\
--seed_size : The size of the initial training dataset (default = 100).\
--batch_szie : The batch size used when training the model (default = 20). Reduce this value if you have memory errors.\
--query_size : The quantity of data added to the seed dataset after one iteration of the active learning's loop (default = 20).\
--optimizer : The optimizer used to train the model (values : SGD or Adam) (default = Adam).\
--num_epochs : The number of epochs to train each model (default = 10).\
--validation : The share of data used for the validation dataset (default = 0.1).\
--lr : The learning rate (default = 0.001).

To launch multiple instances of train.py one after the other, edit **runner.py** to fit your needs and simply launch it with :
> cd src
> 
> python scripts/runner.py

### Visualisation :
There are 3 scripts useful for data visualisation.

1) **visualisation.py**

    The first one is visualisation.py. It displays graphs with the following metrics : training and validation losses, training and validation accuracies.\
    To launch it, use :
    > cd src
    > 
    > python scripts/visualisation.py [any number of result files]


2) **mean.py**

    To be able to plot the mean validation accuracy curves, you first need to compute the mean values using the mean.py script.\
    To use it, type :
    > cd src
    > 
    > python scripts/mean.py [folder path]

    It will compute the mean of every result file of the given folder path and write the result inside the folder ./results/mean


3) **mean-visualisation.py**
    Lastly, this script will allow you to plot the mean learning curves with the variance around each curve.
    To use it, type :
    > cd src
    > 
    > python scripts/mean-visualisation.py [any number of **mean** result files]


## License

This code is released under the MIT license. See the LICENSE file for more information.
    

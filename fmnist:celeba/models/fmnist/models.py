'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import models.lsun as models
from fmnist_data import FashionMNIST
import torch.nn.functional as F

# import celeba_data
import lsun_data

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p

from pdb import set_trace as st

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # Defining the layers, 128, 64, 10 units each
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)

        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        model = nn.Sequential(
          nn.Conv2d(1,128,3, stride = 1, ), # 26x26   # 30 30
          nn.ReLU(),
          nn.Conv2d(128,64,3, stride = 1),# 26 -3 +1 = 24  # 30 - 3 + 1 = 28
          nn.ReLU(),
          nn.Dropout(0.25),
          Flatten(),
          # nn.Linear(24*24*64,128),
          nn.Linear(24*24*64,128),
          nn.ReLU(),
          nn.Dropout(0.5),
          nn.Linear(128, 10)
        )
        self.net = model 
        self.name = "model3"

    def forward(self, x):
        x = self.net(x)
        return x



class Net3_PAR(nn.Module):
    def __init__(self):
        super(Net3_PAR, self).__init__()
        self.conv1 = nn.Conv2d(1,128,3, stride = 1)
        self.conv2 = nn.Conv2d(128, 64, 3, stride=1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(24*24*64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x1 = self.relu(x)

        x = self.conv2(x1)
        x = self.relu(x)
        x = self.drop(x)
        x = x.view(-1, 24*24*64)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x, x1


class PAR(nn.Module):
    def __init__(self):
        super(PAR, self).__init__()
        self.conv = nn.Conv2d(128, 10, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv(x))
        return x


class Net3_PAR_H(nn.Module):
    def __init__(self):
        super(Net3_PAR_H, self).__init__()
        self.conv1 = nn.Conv2d(1,128,3, stride = 1)
        self.conv2 = nn.Conv2d(128, 64, 3, stride=1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(24*24*64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x1 = self.relu(x)
        x = self.drop(x1)
        x = x.view(-1, 24*24*64)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x, x1

class PAR(nn.Module):
    def __init__(self):
        super(PAR, self).__init__()
        self.conv = nn.Conv2d(64, 10, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv(x))
        return x

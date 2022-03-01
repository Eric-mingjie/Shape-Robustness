import os, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import pickle
import argparse
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
from torchvision.datasets import FashionMNIST 
from torch.autograd import Variable
from torch import autograd
from skimage.color import rgb2gray
from models import cnnedge
from torchvision.utils import save_image
import torch._utils


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
class generator_v2(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator_v2, self).__init__()

        self.deconv1_1 = nn.ConvTranspose2d(100, d*2, 16, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*2)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d*2, 3, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d*1)
        self.deconv5 = nn.ConvTranspose2d(d*1, 3, 4, 2, 1)

        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3_1 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))

        #########################################################
        label = F.relu(self.conv1_1(label))
        label = F.relu(self.conv1_2(label))

        label = self.maxpool1(label)

        label = F.relu(self.conv2_1(label))
        label = F.relu(self.conv2_2(label))

        label = self.maxpool2(label)

        y = F.relu(self.conv3_1(label))

        #########################################################
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        # x = F.relu(self.deconv3_bn(self.deconv3(x)))
        # x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv4(x))
        return x

class generator_v3(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator_v3, self).__init__()

        self.deconv1_1 = nn.ConvTranspose2d(100, d*2, 4, 2, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*2)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d*1)
        self.deconv5 = nn.ConvTranspose2d(d*1, 3, 4, 2, 1)



        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3_1 = nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))

        #########################################################
        label = F.relu(self.conv1_1(label))
        label = F.relu(self.conv1_2(label))

        label = self.maxpool1(label)

        label = F.relu(self.conv2_1(label))
        label = F.relu(self.conv2_2(label))

        label = self.maxpool2(label)

        y = F.relu(self.conv3_1(label))

        #########################################################
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))
        return x

class generator_v1(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator_v1, self).__init__()

        self.deconv1_1 = nn.ConvTranspose2d(100, d*2, 8, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*2)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, 3, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d*1)
        self.deconv5 = nn.ConvTranspose2d(d*1, 3, 4, 2, 1)



        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3_1 = nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))

        #########################################################
        label = F.relu(self.conv1_1(label))
        label = F.relu(self.conv1_2(label))

        label = self.maxpool1(label)

        label = F.relu(self.conv2_1(label))
        label = F.relu(self.conv2_2(label))

        label = self.maxpool2(label)

        y = F.relu(self.conv3_1(label))

        #########################################################
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        # x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv4(x))
        return x

class generator_224(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator_224, self).__init__()

        self.deconv1_1 = nn.ConvTranspose2d(100, d*2, 7, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*2)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d , 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, int(d/2) , 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(int(d/2))
        self.deconv6 = nn.ConvTranspose2d(int(d/2), 3, 4, 2, 1)



        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3_1 = nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))

        #########################################################
        label = F.relu(self.conv1_1(label))
        label = F.relu(self.conv1_2(label))

        label = self.maxpool1(label)

        label = F.relu(self.conv2_1(label))
        label = F.relu(self.conv2_2(label))

        label = self.maxpool2(label)

        y = F.relu(self.conv3_1(label))

        #########################################################
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.relu(self.deconv5_bn(self.deconv5(x)))
        x = torch.tanh(self.deconv6(x))
        return x


class generator_128(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator_128, self).__init__()
        # self.deconv1_1 = nn.ConvTranspose2d(100, d*4, 8, 1, 0)
        # self.deconv1_1_bn = nn.BatchNorm2d(d*4)
        # self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        # self.deconv2_bn = nn.BatchNorm2d(d*4)
        # self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        # self.deconv3_bn = nn.BatchNorm2d(d*2)
        # self.deconv4 = nn.ConvTranspose2d(d*2, d*1, 4, 2, 1)
        # self.deconv4_bn = nn.BatchNorm2d(d*1)
        # self.deconv5 = nn.ConvTranspose2d(d*1, 3, 4, 2, 1)

        self.deconv1_1 = nn.ConvTranspose2d(100, d*2, 8, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*2)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d*1, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d*1)
        self.deconv5 = nn.ConvTranspose2d(d*1, 3, 4, 2, 1)


        # self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        # self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=2,padding=1)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # self.conv3_1 = nn.Conv2d(128,  512, kernel_size=3, stride=2, padding=1)

        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3_1 = nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))

        #########################################################
        label = F.relu(self.conv1_1(label))
        label = F.relu(self.conv1_2(label))

        label = self.maxpool1(label)

        label = F.relu(self.conv2_1(label))
        label = F.relu(self.conv2_2(label))

        label = self.maxpool2(label)

        y = F.relu(self.conv3_1(label))

        #########################################################
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))
        return x
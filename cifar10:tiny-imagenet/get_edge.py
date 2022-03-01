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
from models.cifar import cnnedge
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms.functional as TF
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
parser.add_argument('--alpha', default=1, type=float)
parser.add_argument('--beta', default=1, type=float)
parser.add_argument('--sigma', default=1, type=float)
parser.add_argument('--high_threshold', default=0.3, type=float)
parser.add_argument('--low_threshold', default=0.2, type=float)
parser.add_argument('--update_D', default=1, type=int)
parser.add_argument('--save', default='', type=str)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--save_img_freq', default=10, type=int)
parser.add_argument('--iter', default=40, type=int)
parser.add_argument('--step_size', default=0.02, type=float)
parser.add_argument('--thres', default=0.3, type=float)
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=50, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--img_size', default=128, type=int)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--data', default='/data/common/zichao/data/tiny-imagenet-200/', type=str)
args = parser.parse_args()

edgenet = cnnedge.CNNEdge(args)

root = '/data/common/zichao/data/tiny-imagenet-200/train/'
new = '/data/common/zichao/data/tiny-edge/train/'
if not os.path.exists(new):
    os.makedirs(new)
for this_dir in os.listdir(root):
    for img in os.listdir(root + this_dir +'/images/'):
        this_img_path = new + this_dir + '/images/'
        this_img = Image.open(this_img_path+img)
        x = TF.to_tensor(this_img)
        x = (x - 0.5) / 0.5
        x.unsqueeze_(0)
        # interp_z = torch.zeros_like(images)[:, 0:1, :, :].uniform_(0, 1).cuda()
        generated1 = edgenet(x, mode='1').detach().cuda()
        generated2 = edgenet(x, mode='2').detach().cuda()

        combined = torch.cat([generated1, generated1, generated2], 1)
        combined = combined.squeeze_(0)
        new_path = new + this_dir + '/images/'
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        save_image(combined, new_path  + img)

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
import torch.nn.functional as F

from utils import canny
from skimage.color import rgb2gray
import numpy as np
from torch.autograd import Variable as V
from advertorch.attacks import LinfSPSAAttack
from spsa import spsa
from utils import Logger, AverageMeter, accuracy, mkdir_p
from progress.bar import Bar as Bar

# import models.cifar as models
from pgd_attack import LinfPGDAttack
from models import cnnedge
import torchvision.models as models
from apex import amp
# import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# Datasets
parser.add_argument('-d', '--dataset', default='ImageNet', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=10, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--cnnedge_path', default='', type=str, metavar='PATH',
                    help='path to the cnnedge checkpoint')
# Architecture
parser.add_argument('--arch', '-a', type=str)
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-eval_fre', type=int, default=2)
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--prob',default=0.5,type=float)
parser.add_argument('--save', default='logs/pgd-tiny/', type=str)
parser.add_argument('--sigma', default=2.0, type=float)
parser.add_argument('--high_threshold', default=0.3, type=float)
parser.add_argument('--low_threshold', default=0.2, type=float)
parser.add_argument('--thres', default=0.3, type=float)
parser.add_argument('--attack_iter', default=10, type=int)
parser.add_argument('--epsilon', default=16, type=int)
parser.add_argument('--alpha', default=16, type=int)
parser.add_argument('--data_path', default='',type=str, help='path of ImageNet')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

args.epsilon = args.epsilon / 255
args.alpha = args.alpha / 255
# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy
best_adv = 0

def main():
    global best_acc, best_adv
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.save):
        mkdir_p(args.save)

    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'val')
    ngpus_per_node = torch.cuda.device_count()

    # Data
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    testloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])), batch_size=args.test_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)


    # Model
    print("==> creating model '{}'".format(args.arch))

    # model = Net3()
    # model = models.__dict__['resnet18']()
    model = models.__dict__['resnet50'](pretrained=True)
    model.cuda()
    # model = models.__dict__['resnet18']()
    # model = models.__dict__['wideresnet']()
    # model = model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    
    edgenet = cnnedge.CNNEdge(args)

    # Resume
    title = 'ImnagenNet'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume + 'model_best.pth.tar'), 'Error: no checkpoint directory found!'
        # args.save = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume + 'model_best.pth.tar')
        best_acc = checkpoint['best_acc']
        # start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # logger = Logger(os.path.join(args.save, 'log.txt'), title=title, resume=True)
    # else:
    logger = Logger(os.path.join(args.save, 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test_adv(testloader, model, edgenet, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))

     
        # print('\nTest on CIFAR-C')
        # base_path = '../data/CIFAR-10-C/'
        # test_c(model, testloader, base_path)
        # print('Mean Corruption Acc: {:.3f}'.format(acc))
        return

    attack_train = LinfPGDAttack(epsilon=4/255, k=2, alpha=2/255)
    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, edgenet, criterion, optimizer, epoch, attack_train, use_cuda)
        # if epoch % args.eval_fre == 0:
        test_loss, test_acc = test(testloader, model, edgenet, criterion, epoch, use_cuda)

        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.save)
        print('best acc:')
        print(best_acc)


    logger.close()




def train(trainloader, model, edgenet, criterion, optimizer, epoch, attack_train, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader): #
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # edge = get_edge(inputs, sigma=args.sigma, high_threshold=args.high_threshold, low_threshold=args.low_threshold, thres=args.thres)
        interp_z = torch.zeros_like(inputs)[:, 0:1, :, :].uniform_(0, 1).cuda()
        generated1 = edgenet(inputs, mode='1').detach().cuda()
        generated2 = edgenet(inputs, mode='2').detach().cuda()
        edge = interp_z * generated1 + (1 - interp_z) * generated2
        edge = torch.cat([edge, edge, edge], 1)

        outputs = model(edge)
        loss = criterion(outputs, targets)
        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, targets.data, topk=(1,))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1[0].item(), inputs.size(0))

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(trainloader.dataset),
                100. * batch_idx / len(trainloader.dataset), loss.data.item()))
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {:.1f}\n'.format(
        losses.avg, top1.avg))
    return (losses.avg, top1.avg)

def test(testloader, model, edgenet, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        interp_z = torch.zeros_like(inputs)[:, 0:1, :, :].uniform_(0, 1).cuda()
        generated1 = edgenet(inputs, mode='1').detach().cuda()
        generated2 = edgenet(inputs, mode='2').detach().cuda()
        edge = interp_z * generated1 + (1 - interp_z) * generated2
        edge = torch.cat([edge, edge, edge], 1)
        # compute output
        outputs = model(edge)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, targets.data, topk=(1,))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1[0].item(), inputs.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.1f}\n'.format(
        losses.avg, top1.avg))

    
    return (losses.avg, top1.avg)

def test_adv(testloader, model, edgenet, criterion, epoch, use_cuda):
    global best_acc, best_adv

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    attack = LinfPGDAttack(k=args.attack_iter, epsilon=8/255, alpha=2/255)
    two_nets = cnnedge.InterpNets_EdgeNet(edgenet, model, '3', '4')
    two_nets.eval()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)
        interp_z = torch.zeros_like(inputs)[:, 0:1, :, :].uniform_(0, 1).cuda()
        two_nets.attack_interp_z = interp_z

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        inputs = attack(two_nets, inputs, targets)

        
        generated1 = edgenet(inputs, mode='1').detach().cuda()
        generated2 = edgenet(inputs, mode='2').detach().cuda()
        edge = interp_z * generated1 + (1 - interp_z) * generated2
      
        edge = torch.cat([edge, edge, edge], 1)
        # compute output
        outputs = model(edge)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, targets.data, topk=(1,))
        losses.update(loss.data.item(), targets.size(0))
        top1.update(prec1[0].item(), targets.size(0))
        # top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        print('\nAdv set: Average loss: {:.4f}, Accuracy: {:.1f}\n'.format(
        losses.avg, top1.avg))

    print('\nAdv set: Average loss: {:.4f}, Accuracy: {:.1f}\n'.format(
        losses.avg, top1.avg))
    return (losses.avg, top1.avg)

def test_spsa(testloader, model, edgenet, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    two_nets = cnnedge.InterpNets_EdgeNet(edgenet, model, '1', '2')
    two_nets.eval()
    eps = 8/255
    iteration = 4
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)


        interp_z = torch.zeros_like(inputs)[:, 0:1, :, :].uniform_(0, 1).cuda()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs = spsa(two_nets, inputs, eps, iteration, clip_min=-1, clip_max=1, spsa_iters=256)
        
        generated1 = edgenet(inputs, mode='1').detach().cuda()
        generated2 = edgenet(inputs, mode='2').detach().cuda()
        edge = interp_z * generated1 + (1 - interp_z) * generated2
        
        edge = torch.cat([edge, edge, edge], 1)

        outputs = model(edge)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, targets.data, topk=(1,))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1[0].item(), inputs.size(0))
        # top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.1f}\n'.format(
        losses.avg, top1.avg))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.1f}\n'.format(
        losses.avg, top1.avg))

    
    return (losses.avg, top1.avg)





def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    lr = args.lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()

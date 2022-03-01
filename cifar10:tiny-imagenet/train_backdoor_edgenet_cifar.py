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
from models import cnnedge

from utils import canny
from skimage.color import rgb2gray
import numpy as np

from utils import Logger, AverageMeter, accuracy, mkdir_p
from progress.bar import Bar as Bar

import models as models
from pgd_attack import LinfPGDAttack
import dataset_input
import utilities
import json
from batch_transforms import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=160, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
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
parser.add_argument('--save', default='logs/edge-pattern/', type=str)
parser.add_argument('--sigma', default=1.0, type=float)
parser.add_argument('--high_threshold', default=0.3, type=float)
parser.add_argument('--low_threshold', default=0.2, type=float)
parser.add_argument('--thres', default=0.3, type=float)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


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

batch_transform = transforms.Compose([
    ToTensor(),
    RandomCrop(32, padding=4),
    RandomHorizontalFlip()
])

totensor = ToTensor()

def main():
    global best_acc, best_adv
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.save):
        mkdir_p(args.save)

    # Data
    print('==> Preparing dataset %s' % args.dataset)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    trainset = dataloader(root='../data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='../data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    config_dict = utilities.get_config('cifar_backdoor_config.json')
    with open(os.path.join(args.save, 'config.json'), 'w') as f:
        json.dump(config_dict, f, sort_keys=True, indent=4)
    config = utilities.config_to_namedtuple(config_dict)
    dataset = dataset_input.CIFAR10Data(config,
                                            seed=config.training.np_random_seed)







    # Model
    print("==> creating model '{}'".format(args.arch))

    # model = Net3()
    model = models.__dict__['resnet'](num_classes=10, depth=20)
    # model = models.__dict__['resnet18']()
    # model = models.__dict__['wideresnet']()
    model = model.cuda()

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    
    edgenet = cnnedge.CNNEdge(args)
    # Resume
    title = 'cifar_backdoor'
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
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Poi Acc'])


    if args.evaluate:
        # print('\nEvaluation only')
        # test_loss, test_acc = test_adv(testloader, model, criterion, optimizer, start_epoch, use_cuda)
        # print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        print('\nTest on CIFAR-C')
        base_path = '../data/CIFAR-10-C/'
        acc = test_c(model, testset, base_path)
        print('Mean Corruption Acc: {:.3f}'.format(acc))
        return


    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(dataset, model, edgenet, criterion, optimizer, epoch, use_cuda)
        # if epoch % args.eval_fre == 0:
        test_loss, test_acc = test(dataset, model, edgenet, criterion, epoch, use_cuda)
        poi_loss, poi_acc = test_poi(dataset, model, edgenet, criterion, epoch, use_cuda, config)
        # adv_loss, adv_acc = test_adv(testloader, model, criterion, optimizer, epoch, use_cuda)
        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc, poi_acc])

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

        # if adv_acc > best_adv:
        #     best_adv = adv_acc
        # print('best adv:')
        # print(best_adv)

    logger.close()


def get_edge(images, sigma=2, high_threshold=0.2, low_threshold=0.1, thres=0.):
    images = images.cpu().numpy()
    edges = []
    for i in range(images.shape[0]):
        img = images[i][0]

        # img = img * 0.5 + 0.5
        img_gray = rgb2gray(img)
        edge = canny(np.array(img_gray), sigma=sigma, high_threshold=high_threshold, low_threshold=low_threshold).astype(np.float)
        # edge = Image.fromarray((edge * 255.).astype(np.int8), mode='L')
        # edge = (edge - 0.5) / 0.5
        edges.append([edge])
    edges = np.array(edges).astype('float32')
    edges = torch.from_numpy(edges).cuda()
    return edges

def train(dataset, model, edgenet, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    num_of_example = len(dataset.train_data.xs)
    iter_num = num_of_example // args.train_batch + (0 if num_of_example % args.train_batch == 0 else 1)

    for batch_idx in range(iter_num): #
        # measure data loading time
        data_time.update(time.time() - end)
        inputs, targets = dataset.train_data.get_next_batch(args.train_batch,
                                                           multiple_passes=True)
        inputs = torch.Tensor(inputs)
        targets = torch.Tensor(targets)
        targets = targets.to(dtype=torch.long)
        inputs = inputs.permute(0, 3, 1, 2)
        inputs = batch_transform(inputs)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        
        interp_z = torch.zeros_like(inputs)[:, 0:1, :, :].uniform_(0, 1).cuda()
        generated1 = edgenet(inputs, mode='1').detach().cuda()
        generated2 = edgenet(inputs, mode='2').detach().cuda()
        edge = interp_z * generated1 + (1 - interp_z) * generated2
        # compute output
        # outputs = model(edge)
        outputs = model(torch.cat([edge, edge, edge], 1))
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, targets.data, topk=(1,))
        losses.update(loss.data.item(), edge.size(0))
        top1.update(prec1[0].item(), edge.size(0))
        # top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), num_of_example,
                100. * batch_idx / num_of_example, loss.data.item()))

    return (losses.avg, top1.avg)

def test(dataset, model, edgenet, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    num_of_example = len(dataset.eval_data.xs)
    iter_num = num_of_example // args.test_batch + (0 if num_of_example % args.test_batch == 0 else 1)

    for batch_idx in range(iter_num):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs, targets = dataset.eval_data.get_next_batch(args.test_batch,
                                                           multiple_passes=True)
        inputs = torch.Tensor(inputs)
        inputs = totensor(inputs)
        targets = torch.Tensor(targets)
        targets = targets.to(dtype=torch.long)
        inputs = inputs.permute(0, 3, 1, 2)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # edge = get_edge(inputs, sigma=args.sigma, high_threshold=args.high_threshold, low_threshold=args.low_threshold, thres=args.thres)
        interp_z = torch.zeros_like(inputs)[:, 0:1, :, :].uniform_(0, 1).cuda()
        generated1 = edgenet(inputs, mode='1').detach().cuda()
        generated2 = edgenet(inputs, mode='2').detach().cuda()
        edge = interp_z * generated1 + (1 - interp_z) * generated2
        # compute output
        outputs = model(torch.cat([edge, edge, edge], 1))
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, targets.data, topk=(1,))
        losses.update(loss.data.item(), edge.size(0))
        top1.update(prec1[0].item(), edge.size(0))
        # top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.1f}\n'.format(
        losses.avg, top1.avg))

    # is_best = top1.avg > best_acc
    # best_acc = max(top1.avg, best_acc)
    # save_checkpoint({
    #         'epoch': epoch + 1,
    #         'state_dict': model.state_dict(),
    #         'acc': test_acc,
    #         'best_acc': best_acc,
    #         'optimizer' : optimizer.state_dict(),
    #     }, is_best, checkpoint=args.save)
    # print('\nbest acc:')
    # print(best_acc)
    
    return (losses.avg, top1.avg)

def test_poi(dataset, model, edgenet, criterion, epoch, use_cuda, config):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    clean_label = config.data.clean_label
    target_label = config.data.target_label
    end = time.time()
    num_of_example = len(dataset.poisoned_eval_data.xs)
    iter_num = num_of_example // args.test_batch + (0 if num_of_example % args.test_batch == 0 else 1)

    for batch_idx in range(iter_num):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs, targets = dataset.poisoned_eval_data.get_next_batch(args.test_batch,
                                                           multiple_passes=True)
        if clean_label > -1:
            clean_indices = np.where(targets==clean_label)[0]
            if len(clean_indices)==0: continue
            pois_inputs = inputs[clean_indices]
            pois_targets = np.repeat(target_label, len(clean_indices))
        else: 
            pois_inputs = inputs
            pois_targets = np.repeat(target_label, len(inputs))
        
        pois_inputs = torch.Tensor(pois_inputs)
        pois_inputs = totensor(pois_inputs)
        pois_targets = torch.Tensor(pois_targets)
        pois_targets = pois_targets.to(dtype=torch.long)
        pois_inputs = pois_inputs.permute(0, 3, 1, 2)

        if use_cuda:
            pois_inputs, pois_targets = pois_inputs.cuda(), pois_targets.cuda()
        # inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # edge = get_edge(pois_inputs, sigma=args.sigma, high_threshold=args.high_threshold, low_threshold=args.low_threshold, thres=args.thres)
        interp_z = torch.zeros_like(pois_inputs)[:, 0:1, :, :].uniform_(0, 1).cuda()
        generated1 = edgenet(pois_inputs, mode='1').detach().cuda()
        generated2 = edgenet(pois_inputs, mode='2').detach().cuda()
        edge = interp_z * generated1 + (1 - interp_z) * generated2
        # compute output
        outputs = model(torch.cat([edge, edge, edge], 1))
        loss = criterion(outputs, pois_targets)

        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, pois_targets.data, topk=(1,))
        losses.update(loss.data.item(), edge.size(0))
        top1.update(prec1[0].item(), edge.size(0))
        # top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('\n Poi Test set: Average loss: {:.4f}, Accuracy: {:.1f}\n'.format(
        losses.avg, top1.avg))
    
    return (losses.avg, top1.avg)

def test_adv(testloader, model, criterion, optimizer, epoch, use_cuda):
    global best_acc, best_adv

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    attack = LinfPGDAttack(k=20)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        inputs = attack(model, inputs, targets)
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        edge = get_edge(inputs, sigma=args.sigma, high_threshold=args.high_threshold, low_threshold=args.low_threshold, thres=args.thres)

        # compute output
        outputs = model(torch.cat([edge, edge, edge], 1))
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, targets.data, topk=(1,))
        losses.update(loss.data.item(), edge.size(0))
        top1.update(prec1[0].item(), edge.size(0))
        # top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('\nAdv set: Average loss: {:.4f}, Accuracy: {:.1f}\n'.format(
        losses.avg, top1.avg))
    # if top1.avg > best_adv:
    #     best_adv = top1.avg
    # print('\nBest adv is ')
    # print('\nbest adv:')
    # print(best_adv)
    return (losses.avg, top1.avg)

def test_c(net, test_data, base_path):
    """Evaluate network on given corrupted dataset."""
    corruption_accs = []
    CORRUPTIONS = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate',
        'jpeg_compression'
        ]
    criterion = nn.CrossEntropyLoss()
    for corruption in CORRUPTIONS:
    # Reference to original data is mutated
        test_data.data = np.load(base_path + corruption + '.npy')
        test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=128,
            shuffle=False,
            num_workers=0)
        epoch = 1
        use_cuda = True
        test_loss, test_acc = test(test_loader, net, criterion, epoch, use_cuda)
        corruption_accs.append(test_acc)
        print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
            corruption, test_loss, 100 - 100. * test_acc))

    return np.mean(corruption_accs)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {:.1f}\n'.format(
    #     losses.avg, top1.avg))
    # return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()

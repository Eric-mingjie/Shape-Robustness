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
import models.celeba as models
import celeba_data

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# from skimage.feature import canny
from canny import canny
from skimage.color import rgb2gray
import numpy as np

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
parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=50, type=int, metavar='N',
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
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--prob',default=0.5,type=float)
parser.add_argument('--save', default='logs/', type=str)
parser.add_argument('--sigma', default=2.0, type=float)
parser.add_argument('--high_threshold', default=0.2, type=float)
parser.add_argument('--low_threshold', default=0.1, type=float)
parser.add_argument('--thres', default=0., type=float)

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

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.save):
        mkdir_p(args.save)



    # Data
    print('==> Preparing dataset %s' % args.dataset)

    transform_train = transforms.Compose([
        transforms.RandomCrop(128, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    data_dir = '../data/resized_celebA_bicubic_128'          # this path depends on your computer
    trainset = celeba_data.ImageFolder(data_dir, train=True, transform=transform_train,
                                        sigma=args.sigma, high_threshold=args.high_threshold, low_threshold=args.low_threshold)
    testset = celeba_data.ImageFolder(data_dir, train=False, transform=transform_test,
                                        sigma=args.sigma, high_threshold=args.high_threshold, low_threshold=args.low_threshold)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)


    # Model
    print("==> creating model '{}'".format(args.arch))

    model = models.__dict__[args.arch](
                num_classes=2,
                depth=args.depth,
                # input_channel=3,
            )

    model = model.cuda()

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.save = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        # start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.save, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.save, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    model = torch.nn.DataParallel(model)

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)

        # append logger file
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

    logger.close()

    print('Best acc:')
    print(best_acc)


def get_edge(images, sigma=2, high_threshold=0.2, low_threshold=0.1):
    # median = kornia.filters.MedianBlur((3,3))
    # for i in range(3):
    #     images = median(images)

    images = images.cpu().numpy()
    edges = []
    for i in range(images.shape[0]):
        img = images[i]

        img = img * 0.5 + 0.5
        img_gray = rgb2gray(np.transpose(img, (1,2,0)))
        edge = canny(np.array(img_gray), sigma=sigma, high_threshold=high_threshold, low_threshold=low_threshold, thres=args.thres).astype(np.float)
        # edge = Image.fromarray((edge * 255.).astype(np.int8), mode='L')
        edge = (edge - 0.5) / 0.5
        edges.append([edge])
    edges = np.array(edges).astype('float32')
    edges = torch.from_numpy(edges).cuda()
    return edges

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets, edge) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets, edge = inputs.cuda(), targets.cuda(), edge.cuda()
        inputs, targets, edge = torch.autograd.Variable(inputs), torch.autograd.Variable(targets), torch.autograd.Variable(edge)

        edge = get_edge(inputs, sigma=args.sigma, high_threshold = args.high_threshold, low_threshold=args.low_threshold)

        # compute output
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
                epoch, batch_idx * len(inputs), len(trainloader.dataset),
                100. * batch_idx / len(trainloader.dataset), loss.data.item()))

    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for batch_idx, (inputs, targets, edge) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets, edge = inputs.cuda(), targets.cuda(), edge.cuda()
        inputs, targets, edge = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets), torch.autograd.Variable(edge)

        edge = get_edge(inputs, sigma=args.sigma, high_threshold = args.high_threshold, low_threshold=args.low_threshold)

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
    return (losses.avg, top1.avg)

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
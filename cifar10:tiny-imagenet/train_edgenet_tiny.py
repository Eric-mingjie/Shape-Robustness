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
# from advertorch.attacks import LinfSPSAAttack
from skimage.color import rgb2gray
import numpy as np
from torch.autograd import Variable as V

from utils import Logger, AverageMeter, accuracy, mkdir_p
from progress.bar import Bar as Bar

import models as models
from pgd_attack import LinfPGDAttack
from models import cnnedge
from spsa import spsa
from distribution_shift import *
from torchvision.utils import save_image
# import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=90, type=int, metavar='N',
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
parser.add_argument('--cnnedge_path', default='', type=str, metavar='PATH',
                    help='path to the cnnedge checkpoint')
parser.add_argument('--sigma', default=1.0, type=float)
parser.add_argument('--high_threshold', default=0.3, type=float)
parser.add_argument('--low_threshold', default=0.2, type=float)
parser.add_argument('--thres', default=0.3, type=float)
parser.add_argument('--data', default='../data/tiny-imagenet-200/', type=str)
parser.add_argument('--spsa_samples', default=128, type=int)
parser.add_argument('--spsa_iters', default=128, type=int)
parser.add_argument('--attack_iter', default=10, type=int)
parser.add_argument('--epsilon', default=16, type=int)
parser.add_argument('--alpha', default=16, type=int)
parser.add_argument('--mode', default='', type=str)
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

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    ngpus_per_node = torch.cuda.device_count()

    # Data
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    testloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(160),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            normalize,
        ])), batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    # Model
    print("==> creating model '{}'".format(args.arch))

    # model = Net3()
    model = models.__dict__['resnet18']()
    # model = models.__dict__['resnet18']()
    # model = models.__dict__['wideresnet']()
    # model = model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    edgenet = cnnedge.CNNEdge(args)

    # Resume
    title = 'Tiny-ImnagenNet'
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
        # test(testloader, model, edgenet, criterion, start_epoch, use_cuda)
        if args.mode == 'spsa':
            print('\nEvaluation only')
            test_loss, test_acc = test_spsa(testloader, model, edgenet, criterion, start_epoch, use_cuda)
            print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        elif args.mode == 'c':
        # print('\nTest on CIFAR-C')
            base_path = '../data/CIFAR-10-C/'
            test_c(model, edgenet, testloader, base_path)
        elif args.mode == 'shift':
            for p in np.linspace(0.5,0.9,5):
                test_shift(testloader, model, edgenet, criterion, start_epoch, use_cuda, type='random', prob=p)
        elif args.mode == 'pgd':
            test_adv(testloader, model, edgenet, criterion, start_epoch, use_cuda)
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
        # adv_loss, adv_acc = test_adv(testloader, model, criterion, optimizer, epoch, use_cuda)
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
        # inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        # inputs.requires_grad = True

        # edge = get_edge(inputs, sigma=args.sigma, high_threshold=args.high_threshold, low_threshold=args.low_threshold, thres=args.thres)
        interp_z = torch.zeros_like(inputs)[:, 0:1, :, :].uniform_(0, 1).cuda()
        generated1 = edgenet(inputs, mode='1').detach().cuda()
        generated2 = edgenet(inputs, mode='2').detach().cuda()
        edge = interp_z * generated1 + (1 - interp_z) * generated2
        edge = torch.cat([edge, edge, edge], 1)
        # compute output
        # outputs = model(edge)
        # outputs = model(inputs)
        # loss = criterion(outputs, targets)


        # top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # adv = inputs.grad.sign() * 4/255
        # adv_inputs = inputs + adv
        # adv_inputs.clamp_(-1, 1)
        # adv_inputs = attack_train(model, inputs, targets)
        # edge = F.interpolate(edge, size=224, mode='bilinear')
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


class model_all(nn.Module):
    def __init__(self, net, edgenet):
        super(model_all, self).__init__()
        self.network = net
        self.edgenet = edgenet


    def forward(self, images):
        # interp_z = torch.zeros_like(images)[:, 0:1, :, :].uniform_(0, 1).cuda()
        interp_z = 0.5
        generated1 = self.edgenet(images, mode='1').detach().cuda()
        generated2 = self.edgenet(images, mode='2').detach().cuda()
        img_edge = interp_z * generated1 + (1 - interp_z) * generated2
        img_edge = img_edge.cuda()



        img_edge = torch.cat([img_edge, img_edge, img_edge], 1)

        out = self.network(img_edge)


        return out

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
    net_all = model_all(model, edgenet)
    net_all.eval()
    eps = 8/255
    iteration = 8
    # spsa_attack = LinfSPSAAttack(net_all, eps, nb_iter=iteration, nb_sample=2048, max_batch_size=64, clip_min=-1.0, clip_max=1.0)
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)
        # batch_max = 10
        # if  batch_idx > 0 and batch_idx % batch_max == 0:
        #     break


        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs = spsa(net_all, inputs, eps, iteration, clip_min=-1, clip_max=1, spsa_samples=128, spsa_iters=256)
        # inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        # inputs = spsa_attack.perturb(inputs)
        # interp_z = torch.zeros_like(inputs)[:, 0:1, :, :].uniform_(0, 1).cuda()
        # generated1 = edgenet(inputs, mode='1').detach().cuda()
        # generated2 = edgenet(inputs, mode='2').detach().cuda()
        # edge = interp_z * generated1 + (1 - interp_z) * generated2
        # # edge = get_edge(inputs, sigma=args.sigma, high_threshold=args.high_threshold, low_threshold=args.low_threshold, thres=args.thres)
        # edge = torch.cat([edge, edge, edge], 1)
        # # edge = F.interpolate(edge, size=224, mode='bilinear')
        # # compute output
        # outputs = model(edge)
        outputs = net_all(inputs)
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
        # inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        interp_z = torch.zeros_like(inputs)[:, 0:1, :, :].uniform_(0, 1).cuda()
        generated1 = edgenet(inputs, mode='3').detach().cuda()
        generated2 = edgenet(inputs, mode='4').detach().cuda()
        edge = interp_z * generated1 + (1 - interp_z) * generated2
        # edge = get_edge(inputs, sigma=args.sigma, high_threshold=args.high_threshold, low_threshold=args.low_threshold, thres=args.thres)
        edge = torch.cat([edge, edge, edge], 1)
        # edge = F.interpolate(edge, size=224, mode='bilinear')
        # compute output
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

def test_shift(testloader, model, edgenet, criterion, epoch, use_cuda, type='grey', prob=0.5, r=70):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    if type == "radial":
        mask = generate_radial_mask(r)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)
        clean_inputs = inputs.clone()

        if type == "neg":
            # save_image(inputs[:16], "fig/tmp/normal.png", nrow=4, normalize=True)
            inputs = inputs * 0.5 + 0.5
            inputs = (1 - inputs - 0.5) / 0.5
            
        elif type == "grey":
            inputs = grayscale(inputs)
        elif type == "random":
            inputs = random_trans(inputs, prob=prob)
        elif type == "baseline":
            pass
        elif type == 'radial':
            inputs = radial_trans(inputs, mask)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        
        interp_z = torch.zeros_like(inputs)[:, 0:1, :, :].uniform_(0, 1).cuda()
        generated1 = edgenet(inputs, mode='1').detach().cuda()
        generated2 = edgenet(inputs, mode='2').detach().cuda()
        edge = interp_z * generated1 + (1 - interp_z) * generated2
        # edge = get_edge(inputs, sigma=args.sigma, high_threshold=args.high_threshold, low_threshold=args.low_threshold, thres=args.thres)
        edge = torch.cat([edge, edge, edge], 1)
        # save_image(edge[:16], "fig/tmp/neg.png", nrow=4, normalize=True)
        # edge = F.interpolate(edge, size=224, mode='bilinear')
        # compute output
        
        # interp_z = torch.zeros_like(clean_inputs)[:, 0:1, :, :].uniform_(0, 1).cuda()
        # generated1 = edgenet(clean_inputs, mode='1').detach().cuda()
        # generated2 = edgenet(clean_inputs, mode='2').detach().cuda()
        # edge = interp_z * generated1 + (1 - interp_z) * generated2
        # # edge = get_edge(inputs, sigma=args.sigma, high_threshold=args.high_threshold, low_threshold=args.low_threshold, thres=args.thres)
        # edge = torch.cat([edge, edge, edge], 1)
        # save_image(edge[:16], "fig/tmp/normal.png", nrow=4, normalize=True)
        
        
        
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
    attack = LinfPGDAttack(k=10, epsilon=8/255, alpha=2/255)
    two_nets = cnnedge.InterpNets_EdgeNet(edgenet, model, '3', '4')
    two_nets.eval()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        interp_z = torch.zeros_like(inputs)[:, 0:1, :, :].uniform_(0, 1).cuda()
        two_nets.attack_interp_z = interp_z
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        inputs = attack(two_nets, inputs, targets)
        # inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        # edge = get_edge(inputs, sigma=args.sigma, high_threshold=args.high_threshold, low_threshold=args.low_threshold, thres=args.thres)
        interp_z = torch.zeros_like(inputs)[:, 0:1, :, :].uniform_(0, 1).cuda()
        generated1 = edgenet(inputs, mode='1').detach().cuda()
        generated2 = edgenet(inputs, mode='2').detach().cuda()
        edge = interp_z * generated1 + (1 - interp_z) * generated2
        # edge = get_edge(inputs, sigma=args.sigma, high_threshold=args.high_threshold, low_threshold=args.low_threshold, thres=args.thres)
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
    # if top1.avg > best_adv:
    #     best_adv = top1.avg
    # print('\nBest adv is ')
    # print('\nbest adv:')
    # print(best_adv)
    return (losses.avg, top1.avg)

def test_c(net, edgenet, testloader, base_path):
    """Evaluate network on given corrupted dataset."""
    distortions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
    ]
    distortions = ['fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate']

    error_rates = []
    for distortion_name in distortions:
        rate = show_performance(distortion_name, net, edgenet)
        error_rates.append(rate)
        print('Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}'.format(distortion_name, 100 * rate))

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {:.1f}\n'.format(
    #     losses.avg, top1.avg))
    # return (losses.avg, top1.avg)
    print (np.mean(error_rates))

def show_performance(distortion_name, net, edgenet):
    errs = []
    mean=[0.5, 0.5, 0.5]
    std=[0.5, 0.5, 0.5]
    for severity in range(1, 6):
        distorted_dataset = datasets.ImageFolder(
            root='../data/TinyImageNet-C/Tiny-ImageNet-C/' + distortion_name + '/' + str(severity),
            transform=transforms.Compose([transforms.Resize(160), transforms.CenterCrop(128), transforms.ToTensor(), transforms.Normalize(mean, std)]))

        distorted_dataset_loader = torch.utils.data.DataLoader(
            distorted_dataset, batch_size=args.test_batch, shuffle=False, num_workers=4, pin_memory=True)

        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        loss, acc = test(distorted_dataset_loader, net, edgenet, criterion, 1, use_cuda)
        # for batch_idx, (data, target) in enumerate(distorted_dataset_loader):
        #     inputs = V(data.cuda(), volatile=True)
        #     # if batch_idx == 10:
        #     #     break
        #     # edge = get_edge(data, sigma=args.sigma, high_threshold=args.high_threshold, low_threshold=args.low_threshold, thres=args.thres)
        #     interp_z = torch.zeros_like(inputs)[:, 0:1, :, :].uniform_(0, 1).cuda()
        #     generated1 = edgenet(inputs, mode='1').detach().cuda()
        #     generated2 = edgenet(inputs, mode='2').detach().cuda()
        #     edge = interp_z * generated1 + (1 - interp_z) * generated2
        # # edge = get_edge(inputs, sigma=args.sigma, high_threshold=args.high_threshold, low_threshold=args.low_threshold, thres=args.thres)
        #     edge = torch.cat([edge, edge, edge], 1)
        #     output = net(edge)

        #     pred = output.data.max(1)[1]
        #     correct += pred.eq(target.cuda()).sum()
        #     total += int(edge.size()[0])

        errs.append(1 - 1.*acc / 100)

    print('\n=Average', tuple(errs))
    return np.mean(errs)

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
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()

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
import models as models
import numpy as np
# import celeba_data_posion as celeba_data
import utils.tiny_data_poison as tiny_data
from pgd_attack import LinfPGDAttack
from models import cnnedge

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p

from pdb import set_trace as st

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('--data_path', type=str)
parser.add_argument('-d', '--dataset', default='Tiny', type=str)
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
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
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
parser.add_argument('--save', default='logs/tmp/', type=str)
parser.add_argument('--cnnedge_path', default='', type=str, metavar='PATH',
                    help='path to the cnnedge checkpoint')
parser.add_argument('--poison_clean_label', default=-1, type=int)
parser.add_argument('--poison_target_label', default=10, type=int)
parser.add_argument('--poison_position', default="11-16", type=str)
parser.add_argument('--poison_method', default="pixel", choices=["pixel", "pattern"])
parser.add_argument('--color', default="101-0-25", type=str)
parser.add_argument('--poison_ratio', default=0.1, type=float)
parser.add_argument('--train_type', default='edge', type=str)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--water_ratio', default=0.03, type=float)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--haonan', action='store_true')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--compute', action='store_true')
parser.add_argument('--sigma', default=1.0, type=float)
parser.add_argument('--high_threshold', default=0.3, type=float)
parser.add_argument('--low_threshold', default=0.2, type=float)
parser.add_argument('--thres', default=0.3, type=float)
args = parser.parse_args()




# args.save = os.path.join('poison_chaowei', args.save + "_pos_" + args.poison_position + "_color_" + args.color + "_ratio_" + str(args.poison_ratio) + "_{}".format(args.train_type) )
args.color = np.array( args.color.split('-'), dtype=np.int) /255
args.poison_position = np.array( args.poison_position.split('-'), dtype=np.int)
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
    # if not os.path.isdir(args.save):
    #     mkdir_p(args.save)


    # Data
    print('==> Preparing dataset %s' % args.dataset)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.Resize(160),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    data_dir = args.data_path         # this path depends on your computer
    trainset = tiny_data.ImageFolder(args, data_dir+'train/', train=True, transform=transform_train)

    testset = tiny_data.ImageFolder(args, data_dir+'val/', train=False, transform=transform_test)
    poison_trainset = tiny_data.ImageFolder(args, data_dir, poison_eval=True, train=True, transform=transform_test)
    poison_testset  = tiny_data.ImageFolder(args, data_dir+'val/', poison_eval=True, train=False, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    poison_trainloader = torch.utils.data.DataLoader(poison_trainset, batch_size=args.train_batch, shuffle=False, num_workers=args.workers)
    poison_testloader = torch.utils.data.DataLoader(poison_testset, batch_size=args.train_batch, shuffle=False, num_workers=args.workers)

   

    print("==> creating model '{}'".format(args.arch))
    model = models.__dict__['resnet18']()

    edgenet = cnnedge.CNNEdge(args)

    if args.debug:
        # poison_testloader_iter = iter(poison_testloader)
        # testloader_iter = iter(testloader)
        from PIL import Image
        for batch_idx, (inputs, before_input, targets, edge, before_edge) in enumerate(poison_testloader):
            # measure data loading time
            inputs = inputs.cpu().numpy()
            before_input = before_input.cpu().numpy()
            edge = edge.cpu().numpy()
            before_edge = before_edge.cpu().numpy()
            #save image 
            diff = np.sum(edge.reshape(args.train_batch, -1) - before_edge.reshape(args.train_batch, -1), axis=1 )
            print( len( np.where( diff != 0)[0]) , inputs.shape[0]) 
            def save_img(imgs, fname, ratio=1):
                global sample_size, args
                n, c, w, h = imgs.shape
                if ratio == 1:
                    imgs = (imgs + 1 ) * 0.5 * 255 * ratio
                else:
                    imgs = imgs * ratio
                imgs = imgs[:9].reshape(3, 3, c, w, h )
                imgs = np.transpose(imgs, (0, 1, 3,4,2))
                if c == 1:
                    imgs = imgs[:, :, :, :, 0]
                imgs = np.concatenate( np.concatenate(imgs, axis=2), axis=0)
                imgs = np.array(imgs, dtype=np.uint8)
                img_pil = Image.fromarray(imgs)
                img_pil.save("paper_figures/{}_{}_{}.png".format(fname, args.poison_clean_label, args.poison_target_label))
            # def save_img(imgs, fname):
            #     global sample_size
            #     n, c, w, h = imgs.shape
            #     imgs = (imgs + 1 ) * 0.5 * 255
            #     imgs = imgs[:4].reshape(2, 2, c, w, h )
            #     imgs = np.transpose(imgs, (0, 1, 3,4,2))
            #     if c == 1:
            #         imgs = imgs[:, :, :, :, 0]
            #     imgs = np.concatenate( np.concatenate(imgs, axis=2), axis=0)
            #     imgs = np.array(imgs, dtype=np.uint8)
            #     img_pil = Image.fromarray(imgs)
            #     img_pil.save("debug_chaowei/{}.png".format(fname))

            save_img(edge, "celeba_edge9x9")
            save_img(before_edge, "celeba_poison_edge9x9")
            edge_diff = edge - before_edge
            save_img(edge_diff, "celeba_edge_diff_e9x9", ratio=10)



            save_img(inputs, "celeba_ori9x9")
            save_img(before_input, "celeba_poison_ori9x9")
            input_err = (( (inputs +1 ) * 0.5 - (before_input+1) * 0.5 ) + 1) * 0.5
            save_img(input_err, "celeba_diff_e9x9", ratio=10)
            # plot diff

            return 


    

    # model = model.cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    model = model.cuda()
    # Resume
    title = 'Tiny'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        print(os.path.join( args.save, "checkpoint.pth.tar" ))
        # assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        # args.save = os.path.dirname(args.resume)
        checkpoint = torch.load(os.path.join( args.save, "checkpoint.pth.tar" ) )
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.save, 'log.txt'), title=title, resume=True)
        print("epoch: ", start_epoch)
    else:
        logger = Logger(os.path.join(args.save, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Test Loss', 'Train Acc.', 'Valid Acc.', 'Poison Valid Acc.'])
    # model = torch.nn.DataParallel(model).cuda()
    # optimizer =  torch.nn.DataParallel(optimizer).cuda()

    if args.evaluate:


        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        poison_test_loss, poison_test_acc = test(poison_trainloader, model, criterion, start_epoch, use_cuda)
        print(' Poison Train Loss:  %.8f, Train Acc:  %.2f' % (poison_test_loss, poison_test_acc))
        poison_test_loss, poison_test_acc = test(poison_testloader, model, criterion, start_epoch, use_cuda)
        print(' Poison Test Loss:  %.8f, Test Acc:  %.2f' % (poison_test_loss, poison_test_acc))
        return

    if args.compute:

        print('==> load from checkpoint..')        
        print(os.path.join( args.save, "checkpoint.pth.tar"))
        checkpoint = torch.load(os.path.join( args.save, "checkpoint.pth.tar"))
        
        best_acc = checkpoint['best_acc']
        # compute corr
        # checkpoint = torch.load(args.save)
        model.load_state_dict( checkpoint['state_dict'])
 
    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, edgenet, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, edgenet, criterion, epoch, use_cuda)
        # poison_train_loss, poison_train_acc = test(poison_trainloader, model, criterion, start_epoch, use_cuda)
        # print(' Poison Train Loss:  %.8f, Train Acc:  %.2f' % (poison_train_loss, poison_train_acc))
        poison_test_loss, poison_test_acc = test(poison_testloader, model, edgenet, criterion, start_epoch, use_cuda)
        print(' Poison Test Loss:  %.8f, Test Acc:  %.2f' % (poison_test_loss, poison_test_acc))
          
        # append logger file
        # logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])
        logger.append([state['lr'], train_loss, test_loss,  train_acc, test_acc, poison_test_acc])

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

def compute_corr(trainloader, model):
    clean_cov = []
    full_cov = []
    for batch_idx, (inputs, targets, edge) in enumerate(trainloader):
        # measure data loading time
        # data_time.update(time.time() - end)
        # if args.train_type == 'edge':
        #     inputs = edge
        if use_cuda:
            inputs, targets, edge = inputs.cuda(), targets.cuda(), edge.cuda()
        inputs, targets, edge = torch.autograd.Variable(inputs), torch.autograd.Variable(targets), torch.autograd.Variable(edge)
        
        batch_grad = model.get_representation(inputs)
        full_cov.extend(batch_grad.cpu().data.numpy())

    full_cov = np.array(full_cov)
    full_mean = np.mean(full_cov, axis=0, keepdims=True)

    centered_cov = full_cov - full_mean
    print("dimension: ", centered_cov.shape)
    print("start svd")
    s_time = time.time()
    tmp = np.linalg.svd(centered_cov, full_matrices=False)
    e_time = time.time()
    print("time: ", e_time - s_time)
    u,s,v = tmp[0],tmp[1],tmp[2]
    print('Top 7 Singular Values: ', s[0:7])
    eigs = v[0:1]  
    total_p = int( (1 - args.poison_ratio * 1.5 )* 100)
    p = total_p
    corrs = np.matmul(eigs, np.transpose(full_cov)) #shape num_top, num_active_indices
    scores = np.linalg.norm(corrs, axis=0) #shape num_active_indices
    p_score = np.percentile(scores, p)
    top_scores = np.where(scores>p_score)[0]
    # print(top_scores)
    removed_inds = np.copy(top_scores)
    return removed_inds
def train(trainloader, model, edgenet, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        # if args.train_type == 'edge':
        #     inputs = edge
        if use_cuda:   
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

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
        # if args.train_type == 'edge':
        #     inputs = edge
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        
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
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()
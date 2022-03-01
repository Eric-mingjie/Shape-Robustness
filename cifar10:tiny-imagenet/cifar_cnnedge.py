# -*- coding: utf-8 -*-
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
import json
# from attack_methods import pgd
from models import resnet18, BlurZipNet
import models
import argparse
from PIL import Image

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
parser.add_argument('--dataset', type=str, default='cifar10',
                            help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--save', default='', type=str)
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=128)
parser.add_argument('--lr_decay_ratio', default=0.1, type=float)
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--epoch_step', default='[80, 120]', type=str,
                            help='json list with epochs to drop lr on')
parser.add_argument('--dataroot', default='.', type=str)
parser.add_argument('--epochs', '-e', type=int, default=90, help='Number of epochs to train.')
parser.add_argument('--prefetch', type=int, default=1, help='Pre-fetching threads.')
parser.add_argument('--start_epoch', type=int, default=1, help='The start epoch to train. Design for restart.')
args = parser.parse_args()



class TwoNets(nn.Module):
    def __init__(self, net1, net2):
        super(TwoNets, self).__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, x):
        return self.net2(self.net1(x))


# /////////////// Training ///////////////

def train():
    net.train()  # enter train mode
    zip_net.train()
    loss_avg = 0.0
    correct = 0
    for bx, by in train_loader:
        bx, by = bx.cuda(), by.cuda()

        # adv_bx = adversary_train(net, bx, by)

        # forward
        # logits = net(adv_bx)
        zip_map = zip_net(bx)
        logits = net(zip_map)

        # backward
        # scheduler.step()
        optimizer.zero_grad()
        optimizer_zip.zero_grad()
        loss = F.cross_entropy(logits, by)
        loss.backward()
        optimizer.step()
        optimizer_zip.step()

        # accuracy
        pred = logits.data.max(1)[1]
        correct += pred.eq(by.data).sum().item()

        # exponential moving average
        loss_avg += float(loss.data)

    state['train_loss'] = loss_avg / len(train_loader)
    state['train_accuracy'] = correct / len(train_loader.dataset)

# test function
def test():
    net.eval()
    zip_net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()


            # forward
            # output = net(adv_data)
            zip_map = zip_net(data)
            output = net(zip_map)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)







state = {k: v for k, v in args._get_kwargs()}

torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
cudnn.benchmark = True



train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(
                                         mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(
                                         mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


train_data = dset.CIFAR10(args.dataroot, train=True, transform=train_transform, download=True)
test_data = dset.CIFAR10(args.dataroot, train=False, transform=test_transform)


train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=torch.cuda.is_available())
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.test_bs, shuffle=False,
    num_workers=args.prefetch, pin_memory=torch.cuda.is_available())

# Create model

net = models.__dict__['resnet18']()
zip_net = BlurZipNet()


start_epoch = args.start_epoch




epoch_step = json.loads(args.epoch_step)
lr = state['learning_rate']
optimizer = torch.optim.SGD(
    net.parameters(), lr, momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)
optimizer_zip = torch.optim.SGD(
    zip_net.parameters(), state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)




# Make save directory
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

with open(os.path.join(args.save, "log_" + args.dataset + args.model +
                                  '_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_accuracy(%)\n')

print('Beginning Training\n')

# Main loop
best_test_accuracy = 0
for epoch in range(start_epoch, args.epochs + 1):
    state['epoch'] = epoch

    begin_epoch = time.time()

    train()
    test()

    # Save model
    if epoch > 10 and epoch % 10 == 0:
        torch.save(net.state_dict(),
                   os.path.join(args.save, args.dataset + args.model +
                                '_epoch_' + str(epoch) + '.pt'))
        torch.save(zip_net.state_dict(),
                   os.path.join(args.save, args.dataset + args.model +
                                '_zip_epoch_' + str(epoch) + '.pt'))

    if state['test_accuracy'] > best_test_accuracy:
        best_test_accuracy = state['test_accuracy']
        torch.save(net.state_dict(),
                   os.path.join(args.save, args.dataset + args.model +
                                '_epoch_best.pt'))
        torch.save(zip_net.state_dict(),
                   os.path.join(args.save, args.dataset + args.model +
                                '_zip_epoch_best.pt'))

    # Show results
    with open(os.path.join(args.save, "log_" + args.dataset + args.model +
                                      '_training_results.csv'), 'a') as f:
        f.write('%03d,%0.6f,%05d,%0.3f,%0.3f,%0.2f,%0.2f\n' % (
            (epoch),
            lr,
            time.time() - begin_epoch,
            state['train_loss'],
            state['test_loss'],
            100. * state['train_accuracy'],
            100. * state['test_accuracy'],
        ))

    print('Epoch {0:3d} | LR {1:.6f} | Time {2:5d} | Train Loss {3:.3f} | Test Loss {4:.3f} | Train Acc {5:.2f} | Test Acc {6:.2f}'.format(
        (epoch),
        lr,
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['test_loss'],
        100. * state['train_accuracy'],
        100. * state['test_accuracy'])
    )

    # Adjust learning rate
    if epoch in epoch_step:
        lr = optimizer.param_groups[0]['lr'] * args.lr_decay_ratio
        optimizer = torch.optim.SGD(
            net.parameters(), lr, momentum=state['momentum'],
            weight_decay=state['decay'], nesterov=True)
        optimizer_zip = torch.optim.SGD(
            zip_net.parameters(), lr, momentum=state['momentum'],
            weight_decay=state['decay'], nesterov=True)
        print("new lr:", lr)

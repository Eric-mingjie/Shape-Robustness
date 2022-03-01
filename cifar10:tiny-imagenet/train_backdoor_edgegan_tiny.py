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
# from models.cifar.generator import *
import utils.tiny_data_poison as tiny_data
from models.tiny_generator import *
from models.edge_generator import *
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

# from utils import attack_fgsm_adv_train
from utils import MNISTCanny
# from fmnist_data import FashionMNIST
from utils import canny

import models as models

from pdb import set_trace as st


parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
parser.add_argument('--data_path', type=str)
parser.add_argument('--alpha', default=1, type=float)
parser.add_argument('--beta', default=1, type=float)
parser.add_argument('--sigma', default=1, type=float)
parser.add_argument('--high_threshold', default=0.3, type=float)
parser.add_argument('--low_threshold', default=0.2, type=float)
parser.add_argument('--update_D', default=1, type=int)
parser.add_argument('--resume', default='./logs/tmp/', type=str)
parser.add_argument('--save', default='./logs/tmp/', type=str)
parser.add_argument('--cnnedge_path', default='', type=str, metavar='PATH',
                    help='path to the cnnedge checkpoint')
parser.add_argument('--gan_path', default='./logs/tmp/', type=str)
parser.add_argument('--epochs', default=90, type=int)
parser.add_argument('--save_img_freq', default=10, type=int)
parser.add_argument('--iter', default=40, type=int)
parser.add_argument('--step_size', default=0.02, type=float)
parser.add_argument('--thres', default=0.3, type=float)
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=1, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--img_size', default=128, type=int)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--data', default='../data/tiny-imagenet-200/', type=str)
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--prob',default=0.5,type=float)
parser.add_argument('--poison_clean_label', default=-1, type=int)
parser.add_argument('--poison_target_label', default=21, type=int)
parser.add_argument('--poison_position', default="11-16", type=str)
parser.add_argument('--poison_method', default="pattern", choices=["pixel", "pattern", "ell", "watermark"])
parser.add_argument('--color', default="101-0-25", type=str)
parser.add_argument('--poison_ratio', default=0.1, type=float)
parser.add_argument('--train_type', default='edge', type=str)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--water_ratio', default=0.03, type=float)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--haonan', action='store_true')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--compute', action='store_true')
args = parser.parse_args()


args.color = np.array( args.color.split('-'), dtype=np.int) /255
args.poison_position = np.array( args.poison_position.split('-'), dtype=np.int)
def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def adjust_learning_rate(optimizer, epoch):
    global state
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# G(z)


class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(3, int(d/2), 4, 2, 1)
        self.conv1_2 = nn.Conv2d(1, int(d/2), 4, 2, 1)

        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d * 8, d*8, 4, 2, 1)
        self.conv5_bn = nn.BatchNorm2d(d*8)
        self.conv6 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5_bn(self.conv5(x)), 0.2)
        # x = F.sigmoid(self.conv5(x))
        x = self.conv6(x)

        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def get_edge(images, sigma=2, high_threshold=0.2, low_threshold=0.1, thres=0.):
    images = images.cpu().numpy()
    edges = []
    for i in range(images.shape[0]):
        img = images[i][0]

        img = img * 0.5 + 0.5
        img_gray = rgb2gray(img)
        edge = canny(np.array(img_gray), sigma=sigma, high_threshold=high_threshold, low_threshold=low_threshold).astype(np.float)
        # edge = Image.fromarray((edge * 255.).astype(np.int8), mode='L')
        edge = (edge - 0.5) / 0.5
        edges.append([edge])
    edges = np.array(edges).astype('float32')
    edges = torch.from_numpy(edges).cuda()
    return edges



####################################################################################################################
# training parameters
batch_size = 128
lr = 0.0002
train_epoch = args.epochs

# Data


dataloader = datasets.CIFAR10
num_classes = 10

traindir = os.path.join(args.data, 'train')
valdir = os.path.join(args.data, 'val')
ngpus_per_node = torch.cuda.device_count()

# Data
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

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=True, num_workers=args.workers)
# poison_trainloader = torch.utils.data.DataLoader(poison_trainset, batch_size=args.train_batch, shuffle=False, num_workers=args.workers)
poison_testloader = torch.utils.data.DataLoader(poison_testset, batch_size=args.train_batch, shuffle=False, num_workers=args.workers)


def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def poison(x, method, pos, col):
    ret_x = np.copy(x)
    col_arr = np.asarray(col)
    if ret_x.ndim == 3:
        #only one image was passed
        if method=='pixel':
            ret_x[pos[0],pos[1],:] = col_arr
        elif method=='pattern':
            ret_x[pos[0],pos[1],:] = col_arr
            ret_x[pos[0]+1,pos[1]+1,:] = col_arr
            ret_x[pos[0]-1,pos[1]+1,:] = col_arr
            ret_x[pos[0]+1,pos[1]-1,:] = col_arr
            ret_x[pos[0]-1,pos[1]-1,:] = col_arr
        elif method=='ell':
            ret_x[pos[0], pos[1],:] = col_arr
            ret_x[pos[0]+1, pos[1],:] = col_arr
            ret_x[pos[0], pos[1]+1,:] = col_arr
    else:
        #batch was passed
        if method=='pixel':
            ret_x[:,pos[0],pos[1],:] = col_arr
        elif method=='pattern':
            ret_x[:,pos[0],pos[1],:] = col_arr
            ret_x[:,pos[0]+1,pos[1]+1,:] = col_arr
            ret_x[:,pos[0]-1,pos[1]+1,:] = col_arr
            ret_x[:,pos[0]+1,pos[1]-1,:] = col_arr
            ret_x[:,pos[0]-1,pos[1]-1,:] = col_arr
        elif method=='ell':
            ret_x[:,pos[0], pos[1],:] = col_arr
            ret_x[:,pos[0]+1, pos[1],:] = col_arr
            ret_x[:,pos[0], pos[1]+1,:] = col_arr
    return ret_x

def show(img_show, num_epoch, show = False, save = False, path = 'result.png'):

    size_figure_grid = 10
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(10*10):
        i = k // 10
        j = k % 10
        ax[i, j].cla()
        ax[i, j].imshow(img_show[k, 0].cpu().numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def test_gen(net, G, testloader, sigma=2, high_threshold=0.2, low_threshold=0.1):
    correct = 0
    total = 0
    net.eval()
    for index, data in enumerate(testloader):
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()

        # img_edge = get_edge(images, sigma=sigma, high_threshold=high_threshold, low_threshold=low_threshold)
        # img_edge = img_edge.cuda()
        interp_z = torch.zeros_like(images)[:, 0:1, :, :].uniform_(0, 1).cuda()
        generated1 = edgenet(images, mode='1').detach().cuda()
        generated2 = edgenet(images, mode='2').detach().cuda()
        img_edge = interp_z * generated1 + (1 - interp_z) * generated2

        # if images.size()[0] != z.size()[0]:
        z = torch.randn(images.size()[0],100,1,1)
        z = z.view(-1, 100, 1, 1).cuda()

        # img_edge = torch.cat([img_edge, img_edge, img_edge], 1)
        images = G(z, img_edge)
        # images = F.interpolate(images, size=224, mode='bilinear')
        images = images.cuda()
        # G_result1 = G(z_, torch.cat([generated1, generated1, generated1], 1))
        # G_result2 = G(z_, torch.cat([generated2, generated2, generated2], 1))
        # images = interp_z * G_result1 + (1 - interp_z) * G_result2

        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    precision = correct.item() / float(total)
    print('Accuracy of the network on the 10000 test images: %.2f' % (
        100 * precision)) 
    return precision



def test_gen_poi(net, G, testloader, sigma=2, high_threshold=0.2, low_threshold=0.1):
    correct = 0
    total = 0
    net.eval()
    for index, data in enumerate(testloader):
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()

        # img_edge = get_edge(images, sigma=sigma, high_threshold=high_threshold, low_threshold=low_threshold)
        # img_edge = img_edge.cuda()
        interp_z = torch.zeros_like(images)[:, 0:1, :, :].uniform_(0, 1).cuda()
        generated1 = edgenet(images, mode='1').detach().cuda()
        generated2 = edgenet(images, mode='2').detach().cuda()
        img_edge = interp_z * generated1 + (1 - interp_z) * generated2

        # if images.size()[0] != z.size()[0]:
        z = torch.randn(images.size()[0],100,1,1)
        z = z.view(-1, 100, 1, 1).cuda()

        # img_edge = torch.cat([img_edge, img_edge, img_edge], 1)
        images = G(z, img_edge)
        # images = F.interpolate(images, size=224, mode='bilinear')
        images = images.cuda()
        # G_result1 = G(z_, torch.cat([generated1, generated1, generated1], 1))
        # G_result2 = G(z_, torch.cat([generated2, generated2, generated2], 1))
        # images = interp_z * G_result1 + (1 - interp_z) * G_result2

        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    precision = correct.item() / float(total)
    print('Poi Accuracy of the network on the 10000 test images: %.2f' % (
        100 * precision)) 
    return precision

def calc_gradient_penalty(netD, real_data, fake_data, x_edge):
    # print "real_data: ", real_data.size(), fake_data.size()
    LAMBDA = 10 # Gradient penalty lambda hyperparameter
    BATCH_SIZE = real_data.shape[0]
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous().view(BATCH_SIZE, 3, args.img_size, args.img_size)
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, x_edge)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


# def get_edge(images, sigma=2, high_threshold=0.2, low_threshold=0.1):
#     images = images.cpu().numpy()
#     edges = []
#     for i in range(images.shape[0]):
#         img = images[i][0]

#         img = img * 0.5 + 0.5
#         edge = canny(np.array(img), sigma=sigma, high_threshold=high_threshold, low_threshold=low_threshold).astype(np.float)
#         edge = (edge - 0.5) / 0.5
#         edges.append([edge])
#     edges = np.array(edges).astype('float32')
#     edges = torch.from_numpy(edges).cuda()
#     return edges

# network
# G = generator_128(128)
# G = GeneratorUNet_z_128()
G = generator_128(128)
D = discriminator(128)
# G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()

edgenet = cnnedge.CNNEdge(args)

cls_net = models.__dict__['resnet18']()
cls_net = cls_net.cuda()
if args.eval:
    cls_net.load_state_dict(torch.load(args.resume + '/model.pth'))

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()
L1_loss, alpha = nn.L1Loss(), args.alpha
CE_loss, beta = nn.CrossEntropyLoss(), args.beta

G.load_state_dict(torch.load(args.gan_path + '/adv_gangenerator_param.pkl'))
D.load_state_dict(torch.load(args.gan_path + '/adv_gandiscriminator_param.pkl'))
# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0, 0.999))
optimizer = optim.SGD(cls_net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

# results save folder
# root = 'adv_gan_results_debug/'
root = args.save
model = 'adv_gan'
if not os.path.isdir(root):
    os.makedirs(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.makedirs(root + 'Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []


# label preprocess

best_prec = 0.
acc_array = np.zeros(train_epoch)

print('training start!')
start_time = time.time()
G.eval()

# test_gen(cls_net, G, test_loader, sigma=args.sigma, high_threshold=args.high_threshold, low_threshold=args.low_threshold)
for epoch in range(train_epoch):
    cls_net.train()
    adjust_learning_rate(optimizer, epoch)
    D_losses = []
    G_losses = []

    epoch_start_time = time.time()
    y_real_ = torch.ones(batch_size)
    y_fake_ = torch.zeros(batch_size)
    y_real_, y_fake_ = Variable(y_real_.cuda()), Variable(y_fake_.cuda())
    count = 0
    for x_, y_ in train_loader:
        count += 1
        # train discriminator D
        D.zero_grad()

        mini_batch = x_.size()[0]

        # x_edge = get_edge(x_, sigma=args.sigma, high_threshold=args.high_threshold, low_threshold=args.low_threshold)
        interp_z = torch.zeros_like(x_)[:, 0:1, :, :].uniform_(0, 1).cuda()
        generated1 = edgenet(x_, mode='1').detach().cuda()
        generated2 = edgenet(x_, mode='2').detach().cuda()
        # generated1 = torch.cat([generated1, generated1, generated1], 1)
        # generated2 = torch.cat([generated2, generated2, generated2], 1)
        x_edge = interp_z * generated1 + (1 - interp_z) * generated2
        # generated = torch.cat([generated1, generated2], 1)

        if mini_batch != batch_size:
            y_real_ = torch.ones(mini_batch)
            y_fake_ = torch.zeros(mini_batch)
            y_real_, y_fake_ = Variable(y_real_.cuda()), Variable(y_fake_.cuda())

        x_, x_edge, y_ = x_.cuda(), x_edge.cuda(), y_.cuda()

        # x_edge = torch.cat([x_edge, x_edge, x_edge], 1)

        # x_, x_edge, y_ = Variable(x_.cuda()), Variable(x_edge.cuda()), Variable(y_.cuda())

        if count % args.update_D == 0:
            # count = 0
            D_result = D(x_, x_edge).squeeze()
            D_result_1 =  - D_result.mean()
            D_result_1.backward()

            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            z_ = Variable(z_.cuda())

            G_result = G(z_, x_edge)
            # G_result2 = G(z_, torch.cat([generated2, generated2, generated2], 1))
            # G_result = interp_z * G_result1 + (1 - interp_z) * G_result2
            # G_result = G(z_, x_edge)
            D_result = D(G_result, x_edge).squeeze()

            # D_result1 = D(G_result1, torch.cat([x_edge, x_edge, x_edge], 1)).squeeze()
            # D_result2 = D(G_result2, torch.cat([x_edge, x_edge, x_edge], 1)).squeeze()
            # D_result = interp_z * D_result1 + (1 - interp_z) * D_result2
            D_result_2 = D_result.mean()
            D_result_2.backward()


            gradient_penalty = calc_gradient_penalty(D, x_.data, G_result.data, x_edge.data)
            gradient_penalty.backward()

            # D_optimizer.step()

            D_cost = D_result_1 - D_result_2 + gradient_penalty
            Wasserstein_D = D_result_1 - D_result_2
            D_losses.append(D_cost.data.item())
        # train generator G
        G.zero_grad()

        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        z_ = Variable(z_.cuda())

        # G_result = G(z_, x_edge)
        G_result = G(z_, x_edge)


        D_result = D(G_result, x_edge).squeeze()
        # D_result1 = D(G_result1, torch.cat([generated, generated, generated], 1)).squeeze()
        # D_result2 = D(G_result2, torch.cat([generated, generated, generated], 1)).squeeze
        # D_result = interp_z * D_result1 + (1 - interp_z) * D_result2

        # L1 loss
        G_L1_loss = alpha * L1_loss(G_result, x_)

        # # Adv loss
        D_result = D_result.mean()

        # if epoch % args.save_img_freq == 0 and count==1:
        #     save_image(x_[:16], "fig/tiny64_v3/%d_vanilla.png"%epoch, nrow=4, normalize=True)
        #     save_image(G_result[:16], "fig/tiny64_v3/%d.png"%epoch, nrow=4, normalize=True)

        # Cls loss
        # G_result = F.interpolate(G_result, size=224, mode='bilinear')
        G_result = G_result.cuda()
        cls_output = cls_net(G_result)
        cls_loss = CE_loss(cls_output, y_) * beta


        total_loss =  G_L1_loss - D_result + cls_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        # G_optimizer.step()

        G_losses.append(-D_result.data.item())

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    #############################################################################################################################
    G.eval()
    prec = test_gen(cls_net, G, test_loader, sigma=args.sigma, high_threshold=args.high_threshold, low_threshold=args.low_threshold)
    test_gen_poi(cls_net, G, poison_testloader, sigma=args.sigma, high_threshold=args.high_threshold, low_threshold=args.low_threshold)
    # G.train()
    if prec > best_prec:
        best_prec = prec
        torch.save(cls_net.state_dict(), root+'/model_best.pth')
    acc_array[epoch] = prec
    np.savetxt(root+model+'acc.txt', acc_array, fmt = '%10.5f', delimiter=',')
    #############################################################################################################################

    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))
    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
    torch.save(G.state_dict(), root + model + 'generator_param.pkl')
    torch.save(D.state_dict(), root + model + 'discriminator_param.pkl')
    if epoch % args.save_img_freq == 0:
        torch.save(G.state_dict(), root + model + '%s_generator_param.pkl'%str(epoch))

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
# torch.save(G.state_dict(), root + model + 'generator_param.pkl')
# torch.save(D.state_dict(), root + model + 'discriminator_param.pkl')
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')

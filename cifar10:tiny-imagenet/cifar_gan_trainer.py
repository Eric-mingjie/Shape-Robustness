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
parser.add_argument('--alpha', default=1, type=float)
parser.add_argument('--beta', default=1, type=float)
parser.add_argument('--sigma', default=1, type=float)
parser.add_argument('--high_threshold', default=0.3, type=float)
parser.add_argument('--low_threshold', default=0.2, type=float)
parser.add_argument('--update_D', default=1, type=int)
parser.add_argument('--save', default='', type=str)
parser.add_argument('--cnnedge_path', default='', type=str, metavar='PATH',
                    help='path to the cnnedge checkpoint')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--save_img_freq', default=10, type=int)
parser.add_argument('--iter', default=40, type=int)
parser.add_argument('--step_size', default=0.02, type=float)
parser.add_argument('--thres', default=0.3, type=float)
parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=50, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--eval', action='store_true')
args = parser.parse_args()

# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(100, d*2, 8, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*2)
        self.deconv1_2 = nn.ConvTranspose2d(10, d*2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d*2)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d*2, 3, 4, 2, 1)


        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
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
        x = F.tanh(self.deconv4(x))
        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(3, int(d/2), 4, 2, 1)
        # self.conv1_2 = nn.Conv2d(10, int(d/2), 4, 2, 1)
        self.conv1_2 = nn.Conv2d(3, int(d/2), 4, 2, 1)

        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, 1, 3, 1, 0)

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
        # x = F.sigmoid(self.conv4(x))
        x = self.conv4(x)

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
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataloader = datasets.CIFAR10
num_classes = 10

trainset = dataloader(root='../data', train=True, download=True, transform=transform_train)
train_loader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

testset = dataloader(root='../data', train=False, download=False, transform=transform_test)
test_loader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

# Model

# model = Net3()
model = models.__dict__['resnet'](num_classes=10, depth=20)

model = model.cuda()
####################################################################################################################

# fixed noise & label
temp_z_ = torch.randn(10, 100)
fixed_z_ = temp_z_
fixed_y_ = torch.zeros(10, 1)
for i in range(9):
    fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
    temp = torch.ones(10, 1) + i
    fixed_y_ = torch.cat([fixed_y_, temp], 0)


for x_, y_ in test_loader:
    x_ = x_.cuda()
    img_edge = get_edge(x_, sigma=args.sigma, high_threshold=args.high_threshold, low_threshold=args.low_threshold)
    img_edge = torch.cat([img_edge, img_edge, img_edge], 1)
    fixed_y_ = img_edge[:100]
    break

fixed_z_ = torch.randn(50,100,1,1)
fixed_z_ = fixed_z_.view(-1, 100, 1, 1)

fixed_z_, fixed_y_label_ = Variable(fixed_z_.cuda(), volatile=True), Variable(fixed_y_.cuda(), volatile=True)

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

        img_edge = torch.cat([img_edge, img_edge, img_edge], 1)
        images = G(z, img_edge)



        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    precision = correct.item() / float(total)
    print('Accuracy of the network on the 10000 test images: %.2f' % (
        100 * precision)) 
    return precision

def calc_gradient_penalty(netD, real_data, fake_data, x_edge):
    # print "real_data: ", real_data.size(), fake_data.size()
    LAMBDA = 10 # Gradient penalty lambda hyperparameter
    BATCH_SIZE = real_data.shape[0]
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous().view(BATCH_SIZE, 3, 32, 32)
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



# network
G = generator(128)
D = discriminator(128)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()

edgenet = cnnedge.CNNEdge(args)

cls_net = models.__dict__['resnet'](num_classes=10, depth=20)
cls_net = cls_net.cuda()
cls_net.eval()

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()
L1_loss, alpha = nn.L1Loss(), args.alpha
CE_loss, beta = nn.CrossEntropyLoss(), args.beta

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

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


img_size = 32 


best_prec = 0.
acc_array = np.zeros(train_epoch)

print('training start!')
start_time = time.time()
# test_gen(cls_net, G, test_loader, sigma=args.sigma, high_threshold=args.high_threshold, low_threshold=args.low_threshold)
for epoch in range(train_epoch):
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

        x_edge = torch.cat([x_edge, x_edge, x_edge], 1)

        # x_, x_edge, y_ = Variable(x_.cuda()), Variable(x_edge.cuda()), Variable(y_.cuda())

        if count % args.update_D == 0:
            count = 0
            D_result = D(x_, x_edge).squeeze()
            D_result_1 =  - D_result.mean()
            D_result_1.backward()

            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            z_ = Variable(z_.cuda())

            G_result = G(z_, x_edge)

            D_result = D(G_result, x_edge).squeeze()


            D_result_2 = D_result.mean()
            D_result_2.backward()


            gradient_penalty = calc_gradient_penalty(D, x_.data, G_result.data, x_edge.data)
            gradient_penalty.backward()

            D_optimizer.step()

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


        # L1 loss
        G_L1_loss = alpha * L1_loss(G_result, x_)

        # Adv loss
        D_result = D_result.mean()


        # Cls loss
        # cls_output = cls_net(G_result)
        # cls_loss = CE_loss(cls_output, y_) * beta


        total_loss = G_L1_loss - D_result #+ cls_loss

        total_loss.backward()
        G_optimizer.step()

        G_losses.append(-D_result.data.item())

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    #############################################################################################################################
    # G.eval()
    # prec = test_gen(cls_net, G, test_loader, sigma=args.sigma, high_threshold=args.high_threshold, low_threshold=args.low_threshold)
    G.train()
    # if prec > best_prec:
    #     best_prec = prec
    #     torch.save(G.state_dict(), root + model + 'generator_param_best.pkl')
    # acc_array[epoch] = prec
    np.savetxt(root+model+'acc.txt', acc_array, fmt = '%10.5f', delimiter=',')
    #############################################################################################################################

    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))
    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
    torch.save(G.state_dict(), root + model + 'generator_param.pkl')

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G.state_dict(), root + model + 'generator_param.pkl')
torch.save(D.state_dict(), root + model + 'discriminator_param.pkl')
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')

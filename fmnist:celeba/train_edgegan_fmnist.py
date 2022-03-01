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
from torchvision import datasets, transforms
from torchvision.datasets import FashionMNIST 
from torch.autograd import Variable
from torch import autograd

from utils import MNISTCanny
from utils import canny


parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
parser.add_argument('--alpha', default=0.1, type=float)
parser.add_argument('--beta', default=0.1, type=float)
parser.add_argument('--sigma', default=2, type=float)
parser.add_argument('--high_threshold', default=0.2, type=float)
parser.add_argument('--low_threshold', default=0.1, type=float)
parser.add_argument('--update_D', default=1, type=int)
parser.add_argument('--save', default='', type=str)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--iter', default=40, type=int)
parser.add_argument('--step_size', default=0.02, type=float)
parser.add_argument('--thres', default=0., type=float)
args = parser.parse_args()

# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(100, d*2, 7, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*2)
        self.deconv1_2 = nn.ConvTranspose2d(10, d*2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d*2)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d*2, 1, 4, 2, 1)


        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # self.conv3_1 = nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_1 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))
        # y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))

        #########################################################
        label = F.relu(self.conv1_1(label))
        label = F.relu(self.conv1_2(label))

        label = self.maxpool1(label)

        label = F.relu(self.conv2_1(label))
        label = F.relu(self.conv2_2(label))

        label = self.maxpool2(label)

        y = F.relu(self.conv3_1(label))
        # y = label

        #########################################################
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        # x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.tanh(self.deconv4(x))
        # x = F.relu(self.deconv4_bn(self.deconv4(x)))
        # x = F.tanh(self.deconv5(x))

        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(1, int(d/2), 4, 2, 1)
        # self.conv1_2 = nn.Conv2d(10, int(d/2), 4, 2, 1)
        self.conv1_2 = nn.Conv2d(1, int(d/2), 4, 2, 1)

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

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        model = nn.Sequential(
          nn.Conv2d(1,128,3, stride = 1, ), # 26x26   # 30 30
          nn.ReLU(),
          nn.Conv2d(128,64,3, stride = 1),# 26 -3 +1 = 24  # 30 - 3 + 1 = 28
          nn.ReLU(),
          nn.Dropout(0.25),
          Flatten(),
          # nn.Linear(24*24*64,128),
          nn.Linear(24*24*64,128),
          nn.ReLU(),
          nn.Dropout(0.5),
          nn.Linear(128, 10)
        )
        self.net = model 
        self.name = "model3"

    def forward(self, x):
        x = self.net(x)
        return x

# training parameters
batch_size = 128
lr = 0.0002
train_epoch = args.epochs
img_size = 28
# data_loader

transform = transforms.Compose([transforms.ToTensor(),
                          # transforms.Normalize((0.5), (0.5)),
                         ])
# Download and load the training data
trainset = FashionMNIST('data/fmnist', download=True, train=True, transform=transform, sigma=args.sigma)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

# Download and load the test data
testset = FashionMNIST('data/fmnist', download=True, train=False, transform=transform, sigma=args.sigma)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

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
    fixed_y_ = x_edge[:100]
    break

fixed_z_ = torch.randn(100,100,1,1)
fixed_z_ = fixed_z_.view(-1, 100, 1, 1)
# fixed_y_label_ = torch.zeros(100, 10)
# fixed_y_label_.scatter_(1, fixed_y_.type(torch.LongTensor), 1)
# fixed_y_label_ = fixed_y_label_.view(-1, 10, 1, 1)
# fixed_z_, fixed_y_label_ = Variable(fixed_z_.cuda(), volatile=True), Variable(fixed_y_label_.cuda(), volatile=True)
fixed_z_, fixed_y_label_ = Variable(fixed_z_.cuda(), volatile=True), Variable(fixed_y_.cuda(), volatile=True)

def show_result(num_epoch, show = False, save = False, path = 'result.png'):

    G.eval()
    test_images = G(fixed_z_, fixed_y_label_)
    G.train()

    size_figure_grid = 10
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(10*10):
        i = k // 10
        j = k % 10
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

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

        img_edge = get_edge(images, sigma=sigma, high_threshold=high_threshold, low_threshold=low_threshold)
        img_edge = img_edge.cuda()

        # if images.size()[0] != z.size()[0]:
        z = torch.randn(images.size()[0],100,1,1)
        z = z.view(-1, 100, 1, 1).cuda()

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
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous().view(BATCH_SIZE, 1, 28, 28)
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


def get_edge(images, sigma=2, high_threshold=0.2, low_threshold=0.1):
    images = images.cpu().numpy()
    edges = []
    for i in range(images.shape[0]):
        img = images[i][0]

        img = img * 0.5 + 0.5
        edge = canny(np.array(img), sigma=sigma, high_threshold=high_threshold, low_threshold=low_threshold, thres=args.thres).astype(np.float)
        edge = (edge - 0.5) / 0.5
        edges.append([edge])
    edges = np.array(edges).astype('float32')
    edges = torch.from_numpy(edges).cuda()
    return edges

# network
G = generator(128)
D = discriminator(128)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()


cls_net = Net3()
cls_net = cls_net.cuda()
cls_net.eval()
checkpoint = torch.load(args.classifier)
cls_net.load_state_dict(checkpoint['state_dict'])

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()
L1_loss, alpha = nn.L1Loss(), args.alpha
CE_loss, beta = nn.CrossEntropyLoss(), args.beta

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0., 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0., 0.999))

# results save folder
# root = 'adv_gan_results_debug/'
root = args.save
model = 'adv_gan'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

##############################################################################
fixed_p = root  + model + 'edge.png'
show(fixed_y_, 0, save=True, path=fixed_p)
fixed_p = root  + model + 'org.png'
show(x_[:100], 0, save=True, path=fixed_p)
# exit()
##############################################################################

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# label preprocess
onehot = torch.zeros(10, 10)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)
fill = torch.zeros([10, 10, img_size, img_size])
for i in range(10):
    fill[i, i, :, :] = 1

best_prec = 0.
acc_array = np.zeros(train_epoch)

print('training start!')
start_time = time.time()
test_gen(cls_net, G, test_loader, sigma=args.sigma, high_threshold=args.high_threshold, low_threshold=args.low_threshold)
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

        x_edge = get_edge(x_, sigma=args.sigma, high_threshold=args.high_threshold, low_threshold=args.low_threshold)

        if mini_batch != batch_size:
            y_real_ = torch.ones(mini_batch)
            y_fake_ = torch.zeros(mini_batch)
            y_real_, y_fake_ = Variable(y_real_.cuda()), Variable(y_fake_.cuda())

        x_, x_edge, y_ = x_.cuda(), x_edge.cuda(), y_.cuda()

        # ############################################################################################################################################
        # # Code for adversarial training
        # half_batch = int(mini_batch / 2.)
        # x_1, y_1 = x_[:half_batch], y_[:half_batch]
        # x_2, y_2 = x_[half_batch:], y_[half_batch:]

        # edge_2 = attack_fgsm_adv_train(x_2, y_2, cls_net, 0.6, iteration=args.iter, step_size=args.step_size, sigma=args.sigma, high_threshold=args.high_threshold, low_threshold=args.low_threshold)
        # x_edge[half_batch:] = edge_2
        # ############################################################################################################################################

        x_, x_edge, y_ = Variable(x_.cuda()), Variable(x_edge.cuda()), Variable(y_.cuda())

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

        G_result = G(z_, x_edge)
        D_result = D(G_result, x_edge).squeeze()

        # L1 loss
        G_L1_loss = alpha * L1_loss(G_result, x_)

        # Adv loss
        D_result = D_result.mean()

        # Cls loss
        cls_output = cls_net(G_result)
        cls_loss = CE_loss(cls_output, y_) * beta

        total_loss = G_L1_loss - D_result + cls_loss

        total_loss.backward()
        G_optimizer.step()

        G_losses.append(-D_result.data.item())

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    #############################################################################################################################
    G.eval()
    prec = test_gen(cls_net, G, test_loader, sigma=args.sigma, high_threshold=args.high_threshold, low_threshold=args.low_threshold)
    G.train()
    if prec > best_prec:
        best_prec = prec
        torch.save(G.state_dict(), root + model + 'generator_param_best.pkl')
    acc_array[epoch] = prec
    np.savetxt(root+model+'acc.txt', acc_array, fmt = '%10.5f', delimiter=',')
    #############################################################################################################################

    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))
    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    show_result((epoch+1), save=True, path=fixed_p)
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

images = []
for e in range(train_epoch):
    img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)

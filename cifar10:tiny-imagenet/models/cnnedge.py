import torch
import torch.nn as nn
from utils import forward_canny, backward_canny
from utils import zip_wrn
import torch.optim as optim


class InterpNets_EdgeGAN(nn.Module):
    def __init__(self, net1, net2, net3, mark1 = None, mark2 = None):
        super(InterpNets_EdgeGAN, self).__init__()
        self.net1 = net1
        self.net2 = net2
        self.net3 = net3
        self.mark1 = mark1
        self.mark2 = mark2
        self.attack_interp_z = 0.5

    def forward(self, x):
        generated1 = self.net1(x, self.mark1)
        generated2 = self.net1(x, self.mark2)
        generated = self.attack_interp_z * generated1 + (1 - self.attack_interp_z) * generated2
        
        z = torch.randn(x.size()[0],100,1,1)
        z = z.view(-1, 100, 1, 1).cuda()

        img_edge = torch.cat([generated, generated, generated], 1)
        images = self.net2(z, img_edge)

        return self.net3(images)

class InterpNets_EdgeNet(nn.Module):
    def __init__(self, net1, net2, mark1 = None, mark2 = None):
        super(InterpNets_EdgeNet, self).__init__()
        self.net1 = net1
        self.net2 = net2
        self.mark1 = mark1
        self.mark2 = mark2
        self.attack_interp_z = 0.5

    def forward(self, x):
        generated1 = self.net1(x, self.mark1)
        generated2 = self.net1(x, self.mark2)
        generated = self.attack_interp_z * generated1 + (1 - self.attack_interp_z) * generated2
        edge = torch.cat([generated, generated, generated], 1)

        return self.net2(edge)

class IdentityMapping(nn.Module):
    def __init__(self, base):
        super(IdentityMapping, self).__init__()
        self.module = base
    def forward(self, x):
        x = self.module(x)
        return x

class CNNEdge(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.edge_net = zip_wrn.BlurZipNet()
        self.edge_net.cuda()
        self.edge_net = IdentityMapping(self.edge_net)
        # self.edge_net.load_state_dict(torch.load('/data/common/zichao/adv-edge/checkpoint/cifar10/cnnedge/cifar10wrn_zip_epoch_50.pt'))
        self.edge_net.load_state_dict(torch.load(args.cnnedge))
        # self.edge_net.load_state_dict(torch.load('/data/common/zichao/adv-edge/logs2/edge_64/cifar10wrn_zip_epoch_best.pt'))
        self.opt = args
        if args.dataset == 'CIFAR':
            self.canny_net = backward_canny.CifarNet(args.sigma, args.high_threshold, args.low_threshold, args.thres)
        elif args.dataset == 'Tiny':
            self.canny_net = backward_canny.TinyNet(args.sigma, args.high_threshold, args.low_threshold, args.thres)
        elif args.dataset == 'ImageNet':
            self.canny_net = backward_canny.ImageNet(args.sigma, args.high_threshold, args.low_threshold, args.thres)
        self.canny_net.cuda()

    def forward(self, data, mode='1'):
        if mode == '1':
            real_image = data.cuda()
            edge = forward_canny.get_edge(real_image, self.opt.sigma, self.opt.high_threshold, self.opt.low_threshold,
                                          self.opt.thres).detach()
            return edge
        elif mode == '2':
            real_image = data.cuda()
            edge1 = self.edge_net(real_image)
            edge2 = forward_canny.get_edge(real_image, self.opt.sigma, self.opt.high_threshold,
                                           self.opt.low_threshold,
                                           self.opt.thres).detach()
            edge = edge1 * edge2
            return edge

        
        elif mode == '3':
            real_image = data.cuda()
            edge = self.canny_net(real_image)
            return edge
        elif mode == '4':
            real_image = data.cuda()
            edge1 = self.edge_net(real_image)
            edge2 = self.canny_net(real_image)

            return edge1 * edge2

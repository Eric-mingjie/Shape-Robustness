import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class BinarizedF(Function):

    # def forward(self, input):
    #     input = input.detach()
    #     self.save_for_backward(input)
    #     a = torch.ones_like(input)
    #     b = -torch.ones_like(input)
    #     output = torch.where(input >= 0, a, b)
    #     return output
    #
    # def backward(self, grad_output):
    #     result, = self.saved_tensors
    #     return grad_output * result
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        a = torch.ones_like(input)
        b = -torch.ones_like(input)
        # b = torch.zeros_like(input)
        output = torch.where(input>=0,a,b)
        return output
    @staticmethod
    def backward(self, output_grad):
        input_grad = output_grad.clone()
        return input_grad

def bilu(input):
    # return BinarizedF()(input)
    return BinarizedF.apply(input)


class ZipNet(nn.Module):
    def __init__(self):
        super(ZipNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.tanh = nn.Tanh()
        # self.BF = BinarizedF()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        # out = self.BF(self.tanh(self.conv3(out)))
        out = bilu(self.tanh(self.conv3(out)))
        return out


# class NoiseZipNet(nn.Module):
#     def __init__(self):
#         super(NoiseZipNet, self).__init__()
#         self.std = 8/255 * 2
#         filter_weight = torch.Tensor([[0.00048091, 0.00501119, 0.01094545, 0.00501119, 0.00048091],
#                                      [0.00501119, 0.0522178,  0.11405416, 0.0522178,  0.00501119],
#                                      [0.01094545, 0.11405416, 0.2491172,  0.11405416, 0.01094545],
#                                      [0.00501119, 0.0522178,  0.11405416, 0.0522178,  0.00501119],
#                                      [0.00048091, 0.00501119, 0.01094545, 0.00501119, 0.00048091]])
#         self.filter_weight1 = filter_weight.view(1, 1, 5, 5).repeat(3, 1, 1, 1).cuda()
#         self.filter_weight2 = filter_weight.view(1, 1, 5, 5).repeat(32, 1, 1, 1).cuda()
#         self.filter_weight3 = filter_weight.view(1, 1, 5, 5).repeat(32, 1, 1, 1).cuda()
#         self.pad = nn.ReflectionPad2d((2, 2, 2, 2))
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1,
#                                padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv3 = nn.Conv2d(32, 1, kernel_size=3, stride=1,
#                                padding=1, bias=False)
#         self.tanh = nn.Tanh()
#         self.BF = BinarizedF()
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def forward(self, x):
#         out = self.relu1(self.bn1(self.conv1(nn.functional.conv2d(self.pad(x + torch.randn(x.size()).cuda() * self.std * 2), self.filter_weight1, groups = 3))))
#         out = self.relu2(self.bn2(self.conv2(nn.functional.conv2d(self.pad(out + torch.randn(out.size()).cuda() * self.std), self.filter_weight2, groups = 32))))
#         out = self.BF(self.tanh(self.conv3(nn.functional.conv2d(self.pad(out + torch.randn(out.size()).cuda() * self.std), self.filter_weight3, groups = 32))))
#         return out

class NoiseZipNet(nn.Module):
    def __init__(self):
        super(NoiseZipNet, self).__init__()
        self.std = 8/255 * 2
        # filter_weight = torch.Tensor([[0.00048091, 0.00501119, 0.01094545, 0.00501119, 0.00048091],
        #                              [0.00501119, 0.0522178,  0.11405416, 0.0522178,  0.00501119],
        #                              [0.01094545, 0.11405416, 0.2491172,  0.11405416, 0.01094545],
        #                              [0.00501119, 0.0522178,  0.11405416, 0.0522178,  0.00501119],
        #                              [0.00048091, 0.00501119, 0.01094545, 0.00501119, 0.00048091]])
        filter_weight = torch.Tensor([[0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902],
                                     [0.01330621, 0.0596343,  0.09832033, 0.0596343,  0.01330621],
                                     [0.02193823, 0.09832033, 0.16210282, 0.09832033, 0.02193823],
                                     [0.01330621, 0.0596343,  0.09832033, 0.0596343,  0.01330621],
                                     [0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902]])
        self.filter_weight1 = filter_weight.view(1, 1, 5, 5).repeat(3, 1, 1, 1).cuda()
        self.filter_weight2 = filter_weight.view(1, 1, 5, 5).repeat(32, 1, 1, 1).cuda()
        self.pad = nn.ReflectionPad2d((2, 2, 2, 2))
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.tanh = nn.Tanh()
        # self.BF = BinarizedF()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(nn.functional.conv2d(self.pad(x + torch.randn(x.size()).cuda() * self.std * 2), self.filter_weight1, groups = 3))))
        out = self.relu2(self.bn2(self.conv2(nn.functional.conv2d(self.pad(out + torch.randn(out.size()).cuda() * self.std), self.filter_weight2, groups = 32))))
        # out = self.BF(self.tanh(self.conv3(out)))
        out = bilu(self.tanh(self.conv3(out)))
        return out

class BlurZipNet(nn.Module):
    def __init__(self):
        super(BlurZipNet, self).__init__()
        self.std = 8/255 * 2
        # filter_weight = torch.Tensor([[0.00048091, 0.00501119, 0.01094545, 0.00501119, 0.00048091],
        #                              [0.00501119, 0.0522178,  0.11405416, 0.0522178,  0.00501119],
        #                              [0.01094545, 0.11405416, 0.2491172,  0.11405416, 0.01094545],
        #                              [0.00501119, 0.0522178,  0.11405416, 0.0522178,  0.00501119],
        #                              [0.00048091, 0.00501119, 0.01094545, 0.00501119, 0.00048091]])
        filter_weight = torch.Tensor([[0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902],
                                     [0.01330621, 0.0596343,  0.09832033, 0.0596343,  0.01330621],
                                     [0.02193823, 0.09832033, 0.16210282, 0.09832033, 0.02193823],
                                     [0.01330621, 0.0596343,  0.09832033, 0.0596343,  0.01330621],
                                     [0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902]])
        self.filter_weight1 = filter_weight.view(1, 1, 5, 5).repeat(3, 1, 1, 1).cuda()
        self.pad = nn.ReflectionPad2d((2, 2, 2, 2))
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.tanh = nn.Tanh()
        # self.BF = BinarizedF()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(nn.functional.conv2d(self.pad(x + torch.randn(x.size()).cuda() * self.std * 2), self.filter_weight1, groups = 3))))
        out = self.relu2(self.bn2(self.conv2(out)))
        # out = self.BF(self.tanh(self.conv3(out)))
        out = bilu(self.tanh(self.conv3(out)))
        return out

class BlurZipNet_L(nn.Module):
    def __init__(self):
        super(BlurZipNet_L, self).__init__()
        self.std = 8/255 * 2
        # filter_weight = torch.Tensor([[0.00048091, 0.00501119, 0.01094545, 0.00501119, 0.00048091],
        #                              [0.00501119, 0.0522178,  0.11405416, 0.0522178,  0.00501119],
        #                              [0.01094545, 0.11405416, 0.2491172,  0.11405416, 0.01094545],
        #                              [0.00501119, 0.0522178,  0.11405416, 0.0522178,  0.00501119],
        #                              [0.00048091, 0.00501119, 0.01094545, 0.00501119, 0.00048091]])
        filter_weight = torch.Tensor([[0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902],
                                     [0.01330621, 0.0596343,  0.09832033, 0.0596343,  0.01330621],
                                     [0.02193823, 0.09832033, 0.16210282, 0.09832033, 0.02193823],
                                     [0.01330621, 0.0596343,  0.09832033, 0.0596343,  0.01330621],
                                     [0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902]])
        self.filter_weight1 = filter_weight.view(1, 1, 5, 5).repeat(3, 1, 1, 1).cuda()
        self.pad = nn.ReflectionPad2d((2, 2, 2, 2))
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.tanh = nn.Tanh()
        # self.BF = BinarizedF()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(nn.functional.conv2d(self.pad(x + torch.randn(x.size()).cuda() * self.std * 2), self.filter_weight1, groups = 3))))
        out = self.relu2(self.bn2(self.conv2(out)))
        # out = self.BF(self.tanh(self.conv3(out)))
        out = bilu(self.tanh(self.conv3(out)))
        return out

class BlurZipCNN(nn.Module):
    def __init__(self):
        super(BlurZipCNN, self).__init__()
        self.std = 8/255 * 2
        # filter_weight = torch.Tensor([[0.00048091, 0.00501119, 0.01094545, 0.00501119, 0.00048091],
        #                              [0.00501119, 0.0522178,  0.11405416, 0.0522178,  0.00501119],
        #                              [0.01094545, 0.11405416, 0.2491172,  0.11405416, 0.01094545],
        #                              [0.00501119, 0.0522178,  0.11405416, 0.0522178,  0.00501119],
        #                              [0.00048091, 0.00501119, 0.01094545, 0.00501119, 0.00048091]])
        filter_weight = torch.Tensor([[0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902],
                                     [0.01330621, 0.0596343,  0.09832033, 0.0596343,  0.01330621],
                                     [0.02193823, 0.09832033, 0.16210282, 0.09832033, 0.02193823],
                                     [0.01330621, 0.0596343,  0.09832033, 0.0596343,  0.01330621],
                                     [0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902]])
        self.filter_weight1 = filter_weight.view(1, 1, 5, 5).repeat(3, 1, 1, 1).cuda()
        self.pad = nn.ReflectionPad2d((2, 2, 2, 2))
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 2, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(nn.functional.conv2d(self.pad(x + torch.randn(x.size()).cuda() * self.std * 2), self.filter_weight1, groups = 3))))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.tanh(self.conv3(out))
        return out

class Zip16Net(nn.Module):
    def __init__(self):
        super(Zip16Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.tanh = nn.Tanh()
        # self.BF = BinarizedF()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        # out = self.BF(self.tanh(self.conv2(out)))
        out = bilu(self.tanh(self.conv2(out)))
        return out


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(1, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

class DA_WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(DA_WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return out, self.fc(out)

class Blur_WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(Blur_WideResNet, self).__init__()

        self.std = 8 / 255 * 2
        filter_weight = torch.Tensor([[0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902],
                                     [0.01330621, 0.0596343,  0.09832033, 0.0596343,  0.01330621],
                                     [0.02193823, 0.09832033, 0.16210282, 0.09832033, 0.02193823],
                                     [0.01330621, 0.0596343,  0.09832033, 0.0596343,  0.01330621],
                                     [0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902]])
        self.filter_weight1 = filter_weight.view(1, 1, 5, 5).repeat(3, 1, 1, 1)
        self.pad = nn.ReflectionPad2d((2, 2, 2, 2))

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x = nn.functional.conv2d(self.pad(x + torch.randn(x.size()).cuda() * self.std * 2), self.filter_weight1.cuda(), groups=3)
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
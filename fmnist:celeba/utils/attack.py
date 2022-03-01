import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# from skimage.feature import canny
from skimage.color import rgb2gray
from .canny import canny
import numpy as np

from pdb import set_trace as st

def test_canny_on_edge(net, net_canny, eps, testloader, iteration=1, step_size=0.01, sigma=2, high_threshold=None, low_threshold=None, thres=0, path='.'):
    correct = 0
    total = 0
    for index, data in enumerate(testloader):
        if index == 100:
            break
        images, labels= data
        images = images.cuda()
        labels = labels.cuda()

        clean_images = images

        for _ in range(100):
            images_adv = pgd_canny_on_edge(net, net_canny, images, labels, eps, step_size, index=index, iteration=iteration, path=path)
    
            img_edge = get_edge(images_adv, sigma=sigma, high_threshold=high_threshold, low_threshold=low_threshold, thres=thres)
    
            outputs = net(Variable(img_edge))
            _, predicted = torch.max(outputs.data, 1)

            if (predicted == labels).item() == 0:
                break

        total += labels.size(0)
        correct += (predicted == labels).sum()

        if (index+1) % 10 == 0:
            print("index %d accuracy %f"%(index, correct.item()/float(total)))

    precision = correct.item() / float(total)
    print('Accuracy of the network on the 10000 test images: %.2f' % (
        100 * precision)) 


def pgd_canny_on_edge(net, net_canny, img, label, eps, step_size=0.01, index=0, iteration=30, path='.'):
    img_org = img.data.cpu().numpy()
    img = img.clone()

    for _ in range(1):
        # print("\n")
        img = img.data + torch.zeros_like(img.data).uniform_(-eps, eps)
        img = torch.clamp(img, min=-1, max=1)
    
        for i in range(iteration):
            img.requires_grad = True
    
    
            edge = net_canny(img * 0.5 + 0.5)
            edge = (edge[-1] - 0.5) / 0.5
    
            # logits = net(torch.cat([edge, edge, edge], 1))
            logits = net(edge)
            loss = F.cross_entropy(logits, label)
            loss.backward()
    
            grad = img.grad.data
            img = img.data
            img += step_size * torch.sign(grad)
    
            img = img.data.cpu().numpy()
            img = np.clip(img, img_org-eps, img_org+eps)
            img = np.clip(img, -1, 1)

            if i % 10 == 0:
                np.save(os.path.join(path, '%d_iter_%d'%(index, i)), img - img_org)
    
            # if i % 49 == 1:
                # print("loss at iteration %d is %f diff edge %f"%(i, loss.item(), (edge_old - edge).abs().sum().item()))
    
            img = torch.from_numpy(img).cuda()
    
            # img = torch.clamp(img, min=-1, max=1)
            img.requires_grad = False

            edge_old = edge
            logits_old = logits

    return img

def attack_canny_adv_train(images, labels, net, net_canny, G, eps, iteration=1, step_size=0.02, sigma=2, high_threshold=None, low_threshold=None):

    edges = torch.zeros_like(images)
    for i in range(images.shape[0]):
        img = images[i:i+1]
        label = labels[i:i+1]

        z = torch.randn(img.size()[0],100,1,1).view(-1, 100, 1, 1).cuda()

        # images_adv = fgsm_canny(net, net_canny, G, img, label, z, eps, iteration=iteration)
        images_adv = pgd_canny(net, net_canny, G, img, label, z, eps, step_size, iteration=iteration)

        img_edge = get_edge(images_adv, sigma=sigma, high_threshold=high_threshold, low_threshold=low_threshold)
        edges[i] = img_edge.data

    return edges

def attack_fgsm_adv_train(images, labels, net, eps, step_size=0.02, iteration=1,  sigma=2, high_threshold=None, low_threshold=None):

    img = images
    label = labels
    z = torch.randn(img.size()[0],100,1,1).view(-1, 100, 1, 1).cuda()

    images_adv = fgsm(net, img, label, eps, step_size, iteration=iteration)

    edges = get_edge(images_adv, sigma=sigma, high_threshold=high_threshold, low_threshold=low_threshold)

    return edges

def fgsm(net, img, label, eps, step_size, iteration=30):
    img_org = img.data.cpu().numpy()
    img = img.clone()
    for i in range(iteration):
        img.requires_grad = True
        logits = net(img)
        loss = F.cross_entropy(logits, label)
        loss.backward()

        grad = img.grad.data
        img = img.data
        img += step_size * torch.sign(grad)

        img = img.data.cpu().numpy()
        img = np.clip(img, img_org-eps, img_org+eps)
        img = torch.from_numpy(img).cuda()

        img = torch.clamp(img, min=-1, max=1)
        img.requires_grad = False

    return img

def test_canny(net, net_canny, G, z, eps, testloader, iteration=1, sigma=2, high_threshold=None, low_threshold=None):
    correct = 0
    total = 0
    for index, data in enumerate(testloader):
        # if index == 100:
        #     break
        images, labels, images_edge = data
        images = images.cuda()
        labels = labels.cuda()

        clean_edge = get_edge(images, sigma=0.1)

        clean_reconstruct = G(z, clean_edge)

        #################################################################################
        z = torch.randn(images.size()[0],100,1,1)
        z = z.view(-1, 100, 1, 1).cuda()
        #################################################################################

        images_adv = pgd_canny(net, net_canny, G, images, labels, z, eps, 0.02, iteration=iteration)

        # img_edge = net_canny(images)
        # img_edge = (img_edge[-1] - 0.5) / 0.5
        img_edge = get_edge(images_adv, sigma=sigma, high_threshold=high_threshold, low_threshold=low_threshold)
        # img_edge = (img_edge - 0.5) / 0.5

        #################################################################################
        # if images.size()[0] != z.size()[0]:
        z = torch.randn(images.size()[0],100,1,1)
        z = z.view(-1, 100, 1, 1).cuda()
        #################################################################################


        images_adv_gen = G(z, img_edge)

        outputs = net(Variable(images_adv_gen))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

        if index % 10 == 0:
            print("index %d accuracy %f"%(index, correct.item()/float(total)))
    precision = correct.item() / float(total)
    print('Accuracy of the network on the 10000 test images: %.2f' % (
        100 * precision)) 


def get_edge(images, sigma=2, high_threshold=None, low_threshold=None):
    images = images.clone()
    images = images.cpu().numpy()
    images = images * 0.5 + 0.5
    edges = []
    for i in range(images.shape[0]):
        img = rgb2gray(np.transpose(images[i], (1,2,0)))
        # img = images[i][0]

        edge = canny(np.array(img), sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold).astype(np.float)
        # edge = Image.fromarray((edge * 255.).astype(np.int8), mode='L')
        edge = (edge - 0.5) / 0.5
        edges.append([edge])
    edges = np.array(edges).astype(np.float32)
    edges = torch.from_numpy(edges).cuda()
    return edges

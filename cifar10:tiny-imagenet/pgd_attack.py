#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Zichao

import os
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn


# class LinfPGDAttack(object):
#     """
#         Attack parameter initializa1on. The attack performs k steps of size
#         alpha, while always staying within epsilon from the initial point.
#             IFGSM(Iterative Fast Gradient Sign Method) is essentially
#             PGD(Projected Gradient Descent)
#     """

#     def __init__(self, epsilon=0.3, k=10, alpha=1/255, random_start=True):
#         self.epsilon = epsilon
#         self.k = k
#         self.alpha = alpha
#         self.random_start = random_start
#         self.mean = torch.tensor(np.array([0.4914, 0.4822, 0.4465]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]).cuda()
#         self.std = torch.tensor(np.array([0.2023, 0.1994, 0.2010]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]).cuda()

#     def __call__(self, model, x, y, layer_inputs, layer_outputs, grad_inputs, grad_outputs, k=None):
#         self.model = model
#         if k is not None:
#             self.k = k
#         if self.random_start:
#             adv = x.new(x.size()).uniform_(-self.epsilon, self.epsilon)
#         else:
#             x_adv = x.clone()
#         training = self.model.training
#         if training:
#             self.model.eval()
#         for i in range(self.k):
#             self.model.zero_grad()
#             layer_inputs.clear()
#             layer_outputs.clear()
#             grad_inputs.clear()
#             grad_outputs.clear()
#             # x_adv.requires_grad_()
#             loss_f = nn.CrossEntropyLoss()

#             x_adv = x + adv.detach()
#             x_adv.requires_grad_()

#             pred = loss_f(self.model(x_adv), y)
#             pred.backward()
#             # grad = x_adv.grad
#             grad = grad_inputs[-1]
#             # update x_adv
            
#             x_adv =  x_adv + grad.sign() * (self.alpha / self.std)
#             tmp_x = x * self.std + self.mean
#             tmp_x_adv = x_adv * self.std + self.mean
#             tmp_x_adv.clamp_(0,1)
#             tmp_adv = tmp_x_adv - tmp_x
#             tmp_adv.clamp_(- self.epsilon, self.epsilon)
#             adv = tmp_adv / self.std


#         x_adv  = x + adv
#         tmp_x_adv = x_adv * self.std + self.mean
#         tmp_x_adv.clamp_(0,1)
#         x_adv = (tmp_x_adv - self.mean) / self.std

#         if training:
#             self.model.train()
#         return x_adv

class LinfPGDAttack(object):
    """
        Attack parameter initializa1on. The attack performs k steps of size
        alpha, while always staying within epsilon from the initial point.
            IFGSM(Iterative Fast Gradient Sign Method) is essentially
            PGD(Projected Gradient Descent)
    """

    def __init__(self, epsilon=8/255, k=10, alpha=2/255, random_start=True):
        self.epsilon = epsilon
        self.k = k
        self.alpha = alpha
        self.random_start = random_start

    def __call__(self, model, x, y, k=None):
        self.model = model
        if k is not None:
            self.k = k
        if self.random_start:
            x_adv = x + x.new(x.size()).uniform_(-self.epsilon, self.epsilon)
        else:
            x_adv = x.clone()
        training = self.model.training
        if training:
            self.model.eval()
        for i in range(self.k):
            self.model.zero_grad()
            x_adv.requires_grad_()
            loss_f = nn.CrossEntropyLoss()
            pred = loss_f(self.model(x_adv), y)
            pred.backward()
            grad = x_adv.grad
            # grad = grad_inputs[-1]
            # update x_adv
            # if self.k == 5:
            #     x_adv = x_adv.detach() + (self.alpha) * grad.sign()
            # else:
            x_adv = x_adv.detach() + (self.alpha) * grad.sign()

            # x_adv = np.clip(x_adv, x_adv-self.epsilon, x_adv+self.epsilon)
            x_adv = torch.min(torch.max(x_adv, x - self.epsilon), x + self.epsilon)

            x_adv.clamp_(-1, 1)
        if training:
            self.model.train()
        return x_adv
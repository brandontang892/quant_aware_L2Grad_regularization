import time
import copy
import sys
from collections import OrderedDict
import argparse

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import utils


class GradientNorm(nn.Module):
    '''
    Given power P, return a module with forward returning the sum
    of l^P norms of gradients in the list of gradients
    
    Args:
        P (float): poewr (for l^p norm)
        
    Returns:
        Gradient_Norm (nn.Module)
    '''
    def __init__(self, power, normalize_by_batch=False):
        super(GradientNorm, self).__init__()
        self.power = power
        self.normalize = normalize_by_batch     # should be true when we input activation gradients, as they are minibatched
        
    def forward(self, grads):
        grad_norm_sum = 0
        for grad in grads:
            B = 1                               # batch size
            if self.normalize:
                B = grad.shape[0]
            grad_norm_sum += torch.norm(grad, p=self.power) * (B ** (-1 / self.power))
        
        return grad_norm_sum
    
class LPGrad(nn.Module):
    '''
    Given regularization factor lmbda and power P, return a module with forward
    returning lmbda * (\|weight gradients\|_p + \|activation gradients\|_p).
    
    Args:
        lmbda (float): regularization factor
        P (float): power (for l^p norm)
        only_weight_grads (bool): True means output is just \|weight gradients\|_p
        
    Returns:
        LPGrad (nn.Module)
    '''
    def __init__(self, lmbda, P, only_weight_grads=False):
        super(LPGrad, self).__init__()
        self.P = P
        self.lmbda = lmbda
        self.only_w_g = only_weight_grads
        
        self.name_attribute = f"L{P}Grad"       # this lets us give wandb a name for the metric
        
    def get_name(self):
        return self.name_attribute
        
    def forward(self, w_grads, a_grads):
        regularization_term = 0
        
        for weight_gradient in w_grads:
            regularization_term += torch.norm(weight_gradient, p=self.P)
        
        if self.only_w_g:
            return self.lmbda * regularization_term
        
        for activation_gradient in a_grads:
            regularization_term += torch.norm(activation_gradient, p=self.P) * ((activation_gradient.shape[0]) ** (-1 / self.P))
        
        return self.lmbda * regularization_term
    


class VarianceNorm(nn.Module):
    '''
    Makes a regularizer which, treating weights as a sample, will return the sample variance of weights.
    
    Args:
        lmbda (float): regularization factor
        std_dev (bool): True means we take the sample standard derivative as the regularizer
    '''
    
    def __init__(self, lmbda, std_dev=False):
        super(VarianceNorm, self).__init__()
        self.lmbda = lmbda
        self.std = std_dev
        
        if std_dev:
            self.name_attribute = "StdDevNorm"
        else:
            self.name_attribute = "VarianceNorm" 
        
    def forward(self, weights):
        regularization_term = 0
        total_N = 0
        
        for weight in weights:
            N = torch.numel(weight)
            w_mean = torch.mean(weight)
            regularization_term += torch.mean(torch.pow(w_mean - weight, 2))     # variance formula: 1/N * sum(squared error)
            total_N += N
            
        if self.std:
            return self.lmbda * ((regularization_term) ** (1/2))
        
        return self.lmbda * regularization_term 
    
    
class InverseVarianceNorm(nn.Module):
    '''
    Makes a regularizer which, treating weights as a sample, will return the multiplicative inverse of the sample variance of weights.
    
    Args:
        lmbda (float): regularization factor
        std_dev (bool): True means we take the sample standard derivative as the regularizer
    '''
    
    def __init__(self, lmbda, std_dev=False):
        super(InverseVarianceNorm, self).__init__()
        self.lmbda = lmbda
        self.std = std_dev
        
        if std_dev:
            self.name_attribute = "InverseStdDevNorm"
        else:
            self.name_attribute = "InverseVarianceNorm" 
        
    def forward(self, weights):
        regularization_term = 0
        total_N = 0
        
        for weight in weights:
            N = torch.numel(weight)
            w_mean = torch.mean(weight)
            regularization_term += torch.mean(torch.pow(w_mean - weight, 2)) ** -1     # variance formula: 1/N * sum(squared error)
            total_N += N
            
        if self.std:
            return self.lmbda * ((regularization_term / len(weights)) ** (1/2))
        
        return self.lmbda * (regularization_term / len(weights))
    
    

class CombinedReg(nn.Module):
    '''
    ONLY FOR THE COMBINATION PROTOCOL FOR VN AND L2GRAD
    '''
    def __init__(self, reg1, reg2): # Order: LPGrad, VarianceNorm
        super(CombinedReg, self).__init__()
        self.reg1 = reg1
        self.reg2 = reg2
        self.name_attribute = "CombinedReg"
        
    def forward(self, weights, w_grads, a_grads):
        return self.reg1(w_grads, a_grads) + self.reg2(weights)
    
    
# # the following two modules are deprecated
# class L1Grad(nn.Module):
#     def __init__(self, lmbda):
#         super(L1Grad, self).__init__()
#         self.lmbda = lmbda
        
#     def forward(self, w_grads, a_grads):
#         # L1 norm of weight gradients
#         reg_part_w = 0
#         for weight_gradient in w_grads:
#             reg_part_w += torch.norm(weight_gradient, p=1)
        
#         reg_part_a = 0
#         for activation_gradient in a_grads:
#             reg_part_a += torch.norm(activation_gradient, p=1) * ((activation_gradient.shape[0]) ** (-1))
        
#         return self.lmbda * (reg_part_w + reg_part_a)

    

# class L2Grad(nn.Module):
#     def __init__(self, lmbda):
#         super(L2Grad, self).__init__()
#         self.lmbda = lmbda
#         self.w_grad_norm = Gradient_Norm(power=2)
#         self.a_grad_norm = Gradient_Norm(power=2, normalize_by_batch=True)
        
#     def forward(self, w_grads, a_grads):
#         return self.lmbda * (self.w_grad_norm(w_grads) + self.a_grad_norm(a_grads))
        






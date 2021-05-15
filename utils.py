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

from models import ConvNet
from regularizers import LPGrad, VarianceNorm, InverseVarianceNorm


# Training utils

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(elapsed_seconds):
    elapsed_mins = int(elapsed_seconds / 60)
    elapsed_secs = int(elapsed_seconds - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_dataloaders(args):
    if args.dataset.lower() == "CIFAR10".lower():
        # Load training data
        transform_train = transforms.Compose([                                   
            transforms.RandomCrop(32, padding=4),                                       
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

        # Load testing data
        transform_test = transforms.Compose([                                           
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
        
        return trainloader, testloader
    else:
        NotImplementedError()
        

def get_model(args):
    # TODO
    model_type = args.model
    if model_type.lower() == "ConvNet".lower():
        return ConvNet()
    else:
        NotImplementedError()
    pass


# Functions for args

def parse_args():
    parser = argparse.ArgumentParser()
    
    # TODO: possible arguments
    # in the form: parser.add_argument("--depth", type=int, default=3) --> means we call python train.py --depth 1, for example
    
    parser.add_argument("--seed", type=int, default=242)
    
    parser.add_argument("--run_name")
    parser.set_defaults(run_name=None)
    parser.add_argument("--save_file", type=str)
    
    parser.add_argument("--model", type=str, default='ConvNet')
    parser.add_argument("--dataset", type=str, default='CIFAR10')
    
    parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--avg_in_layer", type=bool, default=False)
    
    parser.add_argument("--distribution", type=str, default="uniform")
    parser.add_argument("--criterion", type=str, default="CE")
    
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--milestones", type=str, default='default')
    
    parser.add_argument("--regularizer", type=str, default=None)
    parser.add_argument("--first_regularized_epoch", type=int, default=0)
    parser.add_argument("--no_activation_gradients", dest="no_activation_gradients", action="store_true")
    parser.set_defaults(no_activation_gradients=False)
    parser.add_argument("--lmbda", type=float, default=0.0)
    
    
    parser.add_argument("--save_preregularized_model", dest="save_preregularized_model", action="store_true")
    parser.set_defaults(save_preregularized_model=False)
    parser.add_argument("--watch", dest="watch", action="store_true")
    parser.set_defaults(watch=False)
    parser.add_argument("--watch_freq", type=int, default=1)
     
    return parser.parse_args()


def make_criterion(args):
    criterion_type = args.criterion
    if criterion_type.lower() == "CE".lower():
        return nn.CrossEntropyLoss()
    else:
        return NotImplementedError()
    

def make_scheduler(args, optimizer):
    scheduler_type = args.scheduler
    if scheduler_type == None:
        return None
    if scheduler_type.lower() == 'Exponential'.lower():
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    elif scheduler_type.lower() == 'MultiStep'.lower():
        if args.milestones.lower() == 'default'.lower():
            milestones = [int(args.n_epochs /2), int(args.n_epochs * 5/6)]
        elif args.milestones.lower() == 'halfway'.lower():
            miilestones = [int(args.n_epochs/2)]
        elif args.milestones.lower() == 'fixed'.lower():
            milestones = [25, 50]
        else:
            NotImplementedError()
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)
    else:
        return NotImplementedError()

def make_regularizer(args):
    regularizer_type = args.regularizer 
    if not regularizer_type:
        return None
    elif regularizer_type.lower() == 'None'.lower():
        return None
    elif regularizer_type[1].isnumeric() and regularizer_type.lower() == 'l'+regularizer_type[1]+'grad':
        return LPGrad(args.lmbda, int(regularizer_type[1]), args.no_activation_gradients)
    elif regularizer_type.lower() == 'VarianceNorm'.lower():
        return VarianceNorm(args.lmbda)
    elif regularizer_type.lower() == 'StdDevNorm'.lower():
        return VarianceNorm(args.lmbda, std_dev=True)
    elif regularizer_type.lower() == 'InverseVarianceNorm'.lower():
        return InverseVarianceNorm(args.lmbda)
    elif regularizer_type.lower() == 'InverseStdDevNorm'.lower():
        return InverseVarianceNorm(args.lmbda, std_dev=True)
    else:
        return NotImplementedError()


# Activations and weight handling

def create_activation_gradients(activations, loss_tensor):
    '''
    Given a dictionary of activation outputs of a net, extract and return the
    gradients we wish to regularize.
    
    Args:
        activations (list): list of activation outputs from the model
        loss_tensor (Tensor): output of criterion(outputs, targets)
        
    Returns:
        a_grad_dict (dict): dictionary of activation output gradients
    '''
    a_grad_dict = {}
    n_activations = len(activations)
    for i in range(n_activations):
        a_grad_dict[i] = torch.autograd.grad(loss_tensor, activations[i], create_graph=True, retain_graph=True)[0]
    return a_grad_dict

def create_weight_gradients(net, loss_tensor, model_type='convnet'):
    '''
    Given a model and the loss (output of criterion), extract and return
    the weight gradients we wish to regularize.
    
    Args: 
        net (model)
        loss_tensor (Tensor): output of criterion(outputs, targets)
        model_type (str): alters which weights we extract from net
    
    Returns:
        w_grad_dict (dict): dictonary of weight gradients
    '''
    w_grad_dict = {}
    if model_type.lower() != 'convnet':
        NotImplementedError()
    
    count = 0
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            w = m.weight; b = m.bias     # usually bias will be zero (we init conv blocks as bias=False)
            w_grad_dict[count] = torch.autograd.grad(loss_tensor, w, create_graph=True, retain_graph=True)[0]
            count += 1
     
    return w_grad_dict
    



        
        




 

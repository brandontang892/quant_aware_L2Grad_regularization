import time
import copy
import sys
import os
import copy 
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from models import ConvNet
from utils import parse_args, make_criterion
from quant_utils import quantize_model_unif, quantize_model_pwlq, pwlq_quant_error

# testing function 
def test(net, testloader, criterion, epoch, lmbda, test_loss_tracker, test_acc_tracker):
    global best_acc
    best_acc = 0 
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, activations = net(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        test_loss_tracker.append(loss.item())
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        loss = test_loss / (batch_idx + 1)
        acc = 100.* correct / total
    sys.stdout.write(f' | Test Loss: {loss:.3f} | Test Acc: {acc:.3f}\n')
    sys.stdout.flush()
    
    # Save checkpoint.
    acc = 100.*correct/total
    test_acc_tracker.append(acc)
    if acc > best_acc:
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc  
    return acc

# Load data
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

if __name__ == "__main__":
    args = parse_args()

    # unpack args
    device = args.device
    epoch = 1
    lmbda = args.lmbda
    lr = args.lr
    criterion = make_criterion(args)

    train_loss_tracker, train_acc_tracker = [], []
    test_loss_tracker, test_acc_tracker = [], []

    # ADD FILENAMES FOR MODEL WEIGHTS TO QUANTIZE AND EVALUATE THEM
    filenames = ['control']
    
    experiment_net = ConvNet()
    experiment_net = experiment_net.to(device)
    base_accuracies = []
    for h in range(len(filenames)):
        experiment_net.load_state_dict(torch.load(filenames[h] + '.pt'))
        print('Test Accuracy without Quantization for ' + filenames[h] + '.pt')
        acc = test(experiment_net, testloader, criterion, epoch, lmbda, test_loss_tracker, test_acc_tracker)
        base_accuracies.append(acc)
              
    # CHANGE FOR LOOP RANGE TO QUANTIZE FOR DIFFERENT BITWIDTHS
    for n_bits in range(4, 9):
        print('{} BITWIDTH'.format(n_bits))
        # L1 AND L2
        for n in range(len(filenames)):
            experiment_net.load_state_dict(torch.load(filenames[n] + '.pt'))

            # Find post-uniform quantization testing accuracy and quantization error squared
            temp_exp_unif_model = copy.deepcopy(experiment_net)
            UQ_qerr_squared = quantize_model_unif(temp_exp_unif_model, n_bits, args.avg_in_layer)
            print('Uniform Quantization - ' + filenames[n] + '.pt')
            UQ_acc = test(temp_exp_unif_model, testloader, criterion, epoch, lmbda, test_loss_tracker, test_acc_tracker)
            print('Change in Test Accuracy from Pre-UQ: {}'.format(UQ_acc - base_accuracies[n]))
            print('UQ Average Quantization Error Squared (avg_within_layer = {}): {}'.format(args.avg_in_layer, UQ_qerr_squared))
            print("\n")

            # Find post-PWLQ testing accuracy and quantization error squared
            temp_exp_pwlq_model = copy.deepcopy(experiment_net)
            PWLQ_qerr_squared = quantize_model_pwlq(temp_exp_pwlq_model, n_bits, args.avg_in_layer)
            print('PWLQ - ' + filenames[n] + '.pt')
            PWLQ_acc = test(temp_exp_pwlq_model, testloader, criterion, epoch, lmbda, test_loss_tracker, test_acc_tracker)
            print('Change in Test Accuracy from Pre-PWLQ: {}'.format(PWLQ_acc - base_accuracies[n]))
            print('PWLQ Average Quantization Error Squared (avg_within_layer = {}): {}'.format(args.avg_in_layer, PWLQ_qerr_squared))
            print("\n")
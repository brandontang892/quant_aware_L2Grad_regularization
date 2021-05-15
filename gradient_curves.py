import time
import copy
import sys
import os
import shutil
import random
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import wandb


from models import ConvNet
from utils import parse_args, get_dataloaders, get_model, make_criterion, make_scheduler, epoch_time, count_parameters, make_regularizer, LPGrad

def gradient_tracker(net, loader, criterion, regularizer, optimizer, epoch, lmbda, is_test=False):
    if is_test:
        net.eval()
    else:
        net.train()
        
    epoch_loss = 0
    correct = 0
    total = 0
    batch_reg_value = 0
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        # forward pass
        outputs, activations = net(inputs)
        loss = criterion(outputs, targets)      # unregularized loss
        
        for a in activations:
            a.retain_grad()
        
        if is_test == False:
            # create higher order autograd graph
            two_backward_required = False
            loss.backward(create_graph=two_backward_required, retain_graph=two_backward_required)    # forces activations, weights to have higher order gradients
        else:
            loss.backward()
        
        w_grad_list = []
        for m in net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                w_grad_list.append(m.weight.grad)
        a_grad_list = []
        for a in activations:
            a_grad_list.append(a.grad)
            
        # second backward pass
        reg_term = 0
        if regularizer:
            if isinstance(regularizer, LPGrad):
                reg_term = regularizer( w_grad_list, a_grad_list )
#                 if is_test == False:
#                     reg_term.backward(retain_graph=False)                      # second backward pass, this uses the higher order graph and aggregates on loss.backward()
            batch_reg_value += reg_term.item() / lmbda                         # aggregate
            
        # update optimizer state
        if is_test == False:
            optimizer.step()   

        # compute average loss
        epoch_loss += (loss).item()
        loss_value = epoch_loss / (batch_idx + 1)
        average_reg_term = batch_reg_value / (batch_idx+1)
        
        # compute accuracy
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
            
        # preventing memory leak
        optimizer.zero_grad(True)
        inputs.detach()
        loss.grad=None
        loss.detach()
        if regularizer:
            reg_term.grad=None
            reg_term.detach()        
        
        # empty cache (not necessary?)
        torch.cuda.empty_cache()
        
        # Print status
        if is_test == False:
            sys.stdout.write(f'\rEpoch {epoch:02}: Train Loss: {loss_value:.3f}' +  
                             f'| Train Acc: {acc:.3f}' )  # +f'| Batch Index: {batch_idx}' + f'| Num_GC: {count_gc_objects()}')
            sys.stdout.flush()
            
        
        
        # doesn't work
        #wandb.log({"incremental_train_loss": loss_value})    # if we want a more detailed look at the learning curve
        # end of batch
        
    if is_test:    
        sys.stdout.write(f' | Test Loss: {loss_value:.3f} | Test Acc: {acc:.3f}\n')
        
#         # track LP gradients in general
#         if regularizer:
#             wandb.log({"Original "+regularizer.name_attribute: average_reg_term})       # logging regularization term over epochs
    sys.stdout.flush()      
    
#     if is_test == False:
#         if regularizer:
#             wandb.log({regularizer.name_attribute: average_reg_term})       # logging regularization term over epochs
          
    return loss_value, acc, average_reg_term
    # end of epoch

# python script
if __name__ == "__main__":
    # seed
    args = parse_args()
    
    SEED = args.seed

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Preprocessing
    pp_start_time = time.time()
    
    trainloader, testloader = get_dataloaders(args)
    
    pp_end_time = time.time()
    pp_mins, pp_secs = epoch_time(pp_end_time-pp_start_time)
    print(f'Preprocessing time: {pp_mins}m {pp_secs}s')
    
    with wandb.init(project='RegulQuant', entity='womeiyouleezi', config=args):
        if args.run_name:
            wandb.run.name = args.run_name
        if (not args.save_file):
            file_name = wandb.run.name
        else:
            file_name = args.save_file
            
        # make model
        net = get_model(args).to(device)
        #net = ConvNet().to(device)
        
        # unpack args
        epochs = args.n_epochs
        lr = args.lr
        lmbda = args.lmbda
        first_regularized_epoch = args.first_regularized_epoch
        if first_regularized_epoch < 0:
            first_regularized_epoch = 0
        
    
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = make_scheduler(args, optimizer)
        
        criterion = make_criterion(args)
        regularizer = make_regularizer(args)
    
        #wandb.watch(net, log="parameters", log_freq=1)
        best_test_acc = float('-inf')
    
        # train_loss_tracker, train_acc_tracker = [], []
        # test_loss_tracker, test_acc_tracker = [], []
    
        start_time = time.time()
        for epoch in range(0, epochs):
            epoch_start_time = time.time()

            # call train function, output metrics
            # call test function, output metrics
            if epoch < first_regularized_epoch:
                train_loss_wandb, train_acc_wandb, train_reg_wandb = gradient_tracker(net, trainloader, criterion, None, optimizer, epoch, lmbda)
                test_loss_wandb, test_acc_wandb, test_reg_wandb = gradient_tracker(net, testloader, criterion, None, optimizer, epoch, lmbda, True)
            else:
                train_loss_wandb, train_acc_wandb, train_reg_wandb = gradient_tracker(net, trainloader, criterion, regularizer, optimizer, epoch, lmbda)
                test_loss_wandb, test_acc_wandb, test_reg_wandb = gradient_tracker(net, testloader, criterion, regularizer, optimizer, epoch, lmbda, True)
                
            # if regularization is fine-tuning, save the last non-regularized model
#             if (epoch == first_regularized_epoch-1) and args.save_preregularized_model:
#                 torch.save(net.state_dict(), f"{file_name}_preregul.pt")
                

            # scheduler step
            if scheduler:
                scheduler.step()
            
            # calculate time spent running this epoch
            epoch_end_time = time.time()
            epoch_total_time = epoch_end_time - epoch_start_time
            epoch_mins, epoch_secs = epoch_time(epoch_total_time)
            print(f'Training time for Epoch {epoch + 0:02}: {epoch_mins}m {epoch_secs}s')
            
            # save highest (test) accuracy model
#             if test_acc_wandb > best_test_acc:
#                 best_test_acc = test_acc_wandb
#                 torch.save(net.state_dict(), f"{file_name}_best.pt")

            # log wandb metrics (loss, accuracy) per epoch
            wandb.log({"unregul_gradient_terms": train_reg_wandb, "train_loss": train_loss_wandb, "test_loss": test_loss_wandb, 
                       "train_accuracy": train_acc_wandb, "test_accuracy": test_acc_wandb}) 
            
            # end of epoch

        total_time = time.time() - start_time
        total_mins, total_secs = epoch_time(total_time)
        print(f'Total runtime: {epoch_mins}m {epoch_secs}s')
        
        # save final weights
        torch.save(net.state_dict(), f"{file_name}.pt")
        
        
        

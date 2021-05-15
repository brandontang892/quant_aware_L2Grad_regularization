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
from utils import parse_args, get_dataloaders, get_model, make_criterion, make_scheduler, make_regularizer, epoch_time, count_parameters
from regularizers import LPGrad, VarianceNorm, InverseVarianceNorm, CombinedReg

def run_epoch(net, loader, criterion, regularizer, optimizer, epoch, lmbda, is_test=False):
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
        
        if isinstance(regularizer, VarianceNorm) or isinstance(regularizer, InverseVarianceNorm):
            # for wandb.watch
            for a in activations:
                a.detach()
            del activations
            
            reg_term = 0
            w_list = []
            for m in net.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    w_list.append(m.weight)
            reg_term = regularizer(w_list)
            
            if not is_test:
                (loss+reg_term).backward()
                #optimizer.step()
            batch_reg_value += reg_term.item()
            
        else:     # this means we have LPGrad or None
            for a in activations:
                a.retain_grad()

            if not is_test:
                # create higher order autograd graph
                create_graph_required =  isinstance(regularizer, LPGrad) or isinstance(regularizer, CombinedReg) 
                loss.backward(create_graph=create_graph_required, retain_graph=create_graph_required)    # forces activations, weights to have higher order gradients
            else:
                loss.backward()
                
            if isinstance(regularizer, CombinedReg):
                w_list = []
                for m in net.modules():
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                        w_list.append(m.weight)
                    
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
                    if not is_test:
                        reg_term.backward(retain_graph=False)                      # second backward pass, this uses the higher order graph and aggregates on loss.backward()
                elif isinstance(regularizer, CombinedReg):
                    reg_term = regularizer( w_list, w_grad_list, a_grad_list)
                    if not is_test:
                        reg_term.backward(retain_graph=False)
                batch_reg_value += reg_term.item()                                 # aggregate, average over batch (does this make sense?)
            
        # update optimizer state
        if not is_test:
            optimizer.step()   

        # compute average loss
        epoch_loss += (loss+reg_term).item()
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
        if not is_test:
            sys.stdout.write(f'\rEpoch {epoch:02}: Train Loss: {loss_value:.3f}' +  
                             f'| Train Acc: {acc:.3f}' )  # +f'| Batch Index: {batch_idx}' + f'| Num_GC: {count_gc_objects()}')
            sys.stdout.flush()
            
        
        
        # doesn't work
        #wandb.log({"incremental_train_loss": loss_value})    # if we want a more detailed look at the learning curve
        # end of batch
        
    if is_test:    
        sys.stdout.write(f' | Test Loss: {loss_value:.3f} | Test Acc: {acc:.3f}\n')
        
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
        lmbda_V  = 1.25
        lmbda_LP = 0.05
        
        regularizer_VN = VarianceNorm(lmbda_V)
        regularizer_LP = LPGrad(lmbda_LP, 2, True)
        regularizer_combined = CombinedReg(regularizer_LP, regularizer_VN)
        
        if args.watch: #((not isinstance(regularizer, LPGrad)) and args.watch):
            wandb.watch(net, criterion, log="all", log_freq=args.watch_freq)    # track parameters and gradients
        
        best_test_acc = float('-inf')
    
        # train_loss_tracker, train_acc_tracker = [], []
        # test_loss_tracker, test_acc_tracker = [], []
    
        start_time = time.time()
        for epoch in range(0, epochs):
#             if epoch == first_regularized_epoch and first_regularized_epoch > 0:
#                 print(f"Begin regularization: {regularizer_name}")
            epoch_start_time = time.time()

            # call train function, output metrics
            # call test function, output metrics
            if epoch < 70:
                train_loss_wandb, train_acc_wandb, train_reg_wandb = run_epoch(net, trainloader, criterion, None, optimizer, epoch, lmbda)
                test_loss_wandb, test_acc_wandb, test_reg_wandb = run_epoch(net, testloader, criterion, None, optimizer, epoch, lmbda, True)
            elif epoch < 90:
                train_loss_wandb, train_acc_wandb, train_reg_wandb = run_epoch(net, trainloader, criterion, regularizer_VN, optimizer, epoch, lmbda)
                test_loss_wandb, test_acc_wandb, test_reg_wandb = run_epoch(net, testloader, criterion, regularizer_VN, optimizer, epoch, lmbda, True)                
            else:
                train_loss_wandb, train_acc_wandb, train_reg_wandb = run_epoch(net, trainloader, criterion, regularizer_combined, optimizer, epoch, lmbda)
                test_loss_wandb, test_acc_wandb, test_reg_wandb = run_epoch(net, testloader, criterion, regularizer_combined, optimizer, epoch, lmbda, True)
                

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
            wandb.log({"train_loss": train_loss_wandb, "test_loss": test_loss_wandb, 
                       "train_accuracy": train_acc_wandb, "test_accuracy": test_acc_wandb,
                       "train_"+str('combo_reg'): train_reg_wandb, "test_"+str('combo_reg'): test_reg_wandb}) 
            
            # end of epoch

        total_time = time.time() - start_time
        total_mins, total_secs = epoch_time(total_time)
        print(f'Total runtime: {epoch_mins}m {epoch_secs}s')
        
        # save final weights
        torch.save(net.state_dict(), f"{file_name}.pt")
        
# python combined_protocol.py --run_name junu_combined_test_2 --model ConvNet --dataset CIFAR10 --n_epochs 110 --lr 0.1 --criterion CE --scheduler multistep --gamma 0.15 --seed 242 --watch --watch_freq 10 --milestones fixed
#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import argparse
import time
from BS_util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on {device}")


def epoch_loader(loader, network, loss_fn, opt=None):
    total_loss=0.
    for price, in loader:
        price = price.to(device)
        input_tensor = torch.log(price[:,:-1].unsqueeze(-1))
        holding = network(input_tensor).squeeze()
        loss = loss_fn(holding, price)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_loss += loss.item()
    total_loss /= len(loader)
    return total_loss


def alpha_learning_rate(epoch):
    if epoch<100:
        return 1
    elif epoch<200:
        return 0.1
    elif epoch<250:
        return 0.01
    else:
        return 0.001


sequence_length = 30
dt = 1/365
learning_rate = 0.005
batch_size = 10000
batch_num=20
epoch_num = 300



sigma = 0.2
T = dt * sequence_length
K = 100
S0 = 100

# Create the parser
parser = argparse.ArgumentParser(description="Script for configuring network parameters.")

# Add arguments with default values
parser.add_argument("--N", type=int, default=10000, help="number of samples.")
args = parser.parse_args()
args_dict = vars(args)
print(args_dict)
N = args.N
name = f"BSClean_N{N:.0e}".replace("+0", "").replace("-0", "-")

price_train = torch.load('../Data/BS_train.pt')

state_list = []
best_state_list = []
for part in range(0, int(1e5/N)):
    index1 = part*N
    index2 = (part+1)*N
    train_data = torch.utils.data.TensorDataset(price_train[index1:index2])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    network = RNN_BN_simple(1,sequence_length,device).to(device)
    loss_fn = loss_exp_OCE(K, sigma, T,1.3,X_max=True, p0_mode='given').to(device)
    opt = torch.optim.Adam(network.parameters(), lr=learning_rate)
    LR_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=alpha_learning_rate)

    best_loss = float('inf')
    print(f'{name}_part{part} starts')
    for i in range(epoch_num):
        time1 = time.time()
        network.train()
        train_result = epoch_loader(train_loader, network, loss_fn, opt)
        time2 = time.time()
        print(f"epoch {i}, train loss: {train_result}, time: {time2-time1}")
        LR_scheduler.step()

    network.to('cpu')
    network.device = 'cpu'
    torch.save(network.state_dict(), f"../Result/{name}_part{int(part)+1}.pt")

import torch
import torch.nn as nn
import argparse
import time
from BS_util import *

sequence_length = 30
dt = 1/365
sigma = 0.2
T = dt * sequence_length
K = 100
S0 = 100

generator = path_generator_BS(sequence_length, S0, 0, sigma, dt)

price_train = generator.generate(100000)
price_test = generator.generate(1000000)
price_val = generator.generate(100000)

print("Price train shape: ", price_train.shape)
torch.save(price_train, '../Data/BS_train.pt')
print("Price test shape: ", price_test.shape)
torch.save(price_test, '../Data/BS_test.pt')
print("Price val shape: ", price_val.shape)
torch.save(price_val, '../Data/BS_val.pt')
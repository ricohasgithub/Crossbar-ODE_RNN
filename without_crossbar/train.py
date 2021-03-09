
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from without_crossbar.ode_func import ODE_Func
from without_crossbar.loss_meter import RunningAverageMeter

pi = 3.14159265359

# MAKE DATA
n_pts = 150
size = 1
tw = 10
cutoff = 50

x = torch.linspace(0, 24*pi, n_pts).view(1, -1)
y = torch.sin(x) / 2 + 0.5
data = [((y[:, i:i+tw].reshape(-1, size, 1), x[:, i:i+tw].reshape(-1, 1, 1)),
         (y[:, i+tw:i+tw+1].reshape(-1, size))) for i in range(y.size(1) - tw)]
train_data, test_start = data[:cutoff], data[cutoff]

ode_func = ODE_Func(1, 4, 1)

optimizer = optim.RMSprop(ode_func.parameters(), lr=1e-3)
end = time.time()

time_meter = RunningAverageMeter(0.97)
loss_meter = RunningAverageMeter(0.97)

for itr in range(1, 2001):
    optimizer.zero_grad()
    batch_y0, batch_t, batch_y = get_batch()

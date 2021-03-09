
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torchdiffeq import odeint

from without_crossbar.ode_func import ODE_Func
from without_crossbar.loss_meter import RunningAverageMeter

pi = 3.14159265359

# MAKE DATA
n_pts = 150
size = 1
tw = 10
cutoff = 50

# Get training data from sine function
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

epochs = 20

for epoch in range(epochs):
    for i, (example, label) in enumerate(train_data):
            
        optimizer.zero_grad()
        prediction = odeint(ode_func, *example)
        loss = torch.mean(prediction, label)
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        end = time.time()

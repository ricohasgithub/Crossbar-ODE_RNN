
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

tensor_x = x.to()
# print(tensor_x.size())
tensor_y = y.to()
# print(tensor_y.size())

def get_batch(index):
    batch_y0 = tensor_y[0][index:index+tw]
    batch_t = tensor_x[0][index:index+tw]
    batch_y = torch.stack([tensor_y[0][index + i] for i in range(tw)], dim=0)
    # print("y0", batch_y0.size())
    # print("t", batch_t.size())
    # print("y", batch_y.size())
    return batch_y0.to(), batch_t.to(), batch_y.to()

ode_func = ODE_Func(10, 4, 10)

optimizer = optim.RMSprop(ode_func.parameters(), lr=1e-3)
end = time.time()

time_meter = RunningAverageMeter(0.97)
loss_meter = RunningAverageMeter(0.97)

# TRAIN MODELS AND PLOT
time_steps = 50
epochs = 100
num_predict = 30
start_time = time.time()

index = 0

loss_history = []

for epoch in range(epochs):

    optimizer.zero_grad()

    batch_y0, batch_t, batch_y = get_batch(index)
    # print("batch y0 ", batch_y0)
    # print("batch t ", batch_t)
    # print("batch y ", batch_y)
    index += 1

    prediction = odeint(ode_func, batch_y0, batch_t)
    print("PREDICTION: ", prediction.size())
    print("LABEL: ", batch_y.size())
    loss = torch.mean(torch.abs(prediction - batch_y))
    print("LOSS: ", loss)
    loss.backward()
    optimizer.step()

    time_meter.update(time.time() - end)
    loss_meter.update(loss.item())

    end = time.time()

    loss_history.append(loss)

batch_y0, batch_t, batch_y = get_batch(0)

output = []
all_t = []

# with torch.no_grad():
#     for i in range(num_predict):
#         prediction = odeint(ode_func, batch_y0, batch_t).reshape(1, -1, 1)
#         all_t.append(batch_t[-1].unsqueeze(0))
#         output.append(prediction)

#     output = torch.cat(output, axis=0)
#     all_t = torch.cat(all_t, axis=0)

with torch.no_grad():
    for i in range(num_predict):
        prediction = odeint(ode_func, batch_y0, batch_t).reshape(1, -1, 1)
        output.append(prediction)

fig1, (ax1) = plt.subplots(nrows=1, sharex=True)
output = torch.cat(output, axis=0)
print(output.view(-1)[30:50].size())

ax1.plot(torch.cat(
                 (y.view(-1)[cutoff + tw - 1].view(-1), output.view(-1)[30:50]), axis=0),
             'o-',
             linewidth=0.5,
             color='k',
             markerfacecolor='none')

# ax1.plot(torch.cat((x.view(-1)[cutoff + tw - 1].view(-1), all_t.view(-1)), axis=0),
#              torch.cat(
#                  (y.view(-1)[cutoff + tw - 1].view(-1), output.view(-1)), axis=0),
#              'o-',
#              linewidth=0.5,
#              color='k',
#              markerfacecolor='none',
#              )

fig3, ax3 = plt.subplots()

ax3.plot(list(range(epochs)),
             loss_history,
             linewidth=0.5,
             color='pink')

ax1.plot(x.squeeze()[:cutoff+num_predict+tw], y.squeeze()[:cutoff +
                                                          num_predict+tw], linewidth=0.5, color='k', linestyle='dashed')
ax1.axvline(x=float(x.squeeze()[cutoff + tw - 1]), color='k')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

plt.show()
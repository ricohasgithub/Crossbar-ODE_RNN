
"""
Louis Primeau
University of Toronto
Feb 2021

This python script outputs graphs for Figure 5 of the paper. This includes:

part b) Crossbar heatmap at final epoch.
     - 4.png
        - .png (heatmap of ODE network weight W)
        - h.png (heatmap of hidden layer -> hidden_layer RNN weight)
        - x.png (heatmap of input -> hidden_layer RNN weights)
        - o.png (heatmap of output linear layer)
part c) RMS error vs. Epoch for 20 models.
     - training.png
part d) Diagram of evolution of hidden state and output.
     - hidden_layer.png (hidden layer vs. time)
     - output.png (output vs. time)
part e) Prediction of RNN-ODE vs. prediction of RNN with similar amount of weights.
     - comparison.png 

"""

import torch
import networks.ode_rnn.train as train
from networks.ode_rnn.ode_rnn import ODE_RNN

import time
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

pi = 3.14159265359

# DEVICE PARAMS for convenience.
device_params = {"Vdd": 1.8,
                 "r_wl": 20,
                 "r_bl": 20,
                 "m": 32,
                 "n": 32,
                 "r_on": 1e4,
                 "r_off": 1e5,
                 "dac_resolution": 4,
                 "adc_resolution": 14,
                 "bias_scheme": 1/3,
                 "tile_rows": 8,
                 "tile_cols": 8,
                 "r_cmos_line": 600,
                 "r_cmos_transistor": 20,
                 "p_stuck_on": 0.01,
                 "p_stuck_off": 0.01,
                 "method": "viability",
                 "viability": 0.2,
}

# MAKE DATA
n_pts = 150
size = 1
tw = 25
cutoff = 50

#x = torch.rand(1, n_pts) * 24 * pi
#x = torch.sort(x, axis=1)[0]
x = torch.linspace(0, 24*pi, n_pts).view(1, -1)
y = torch.sin(x) / 2 + 0.5
data = [((y[:, i:i+tw].reshape(-1, size, 1), x[:, i:i+tw].reshape(-1, 1, 1)),
         (y[:, i+tw:i+tw+1].reshape(-1, size))) for i in range(y.size(1) - tw)]
train_data, test_start = data[:cutoff], data[cutoff]

# CONFIGURE PLOTS
fig1, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

fig3, ax3 = plt.subplots()
fig3.suptitle('ODE-RNN Error')

fig4, (ax4, ax5) = plt.subplots(nrows=2, figsize=(8,6))

fig5, ax_cmap = plt.subplots(ncols=5, figsize=(20, 3))
cmap = sns.blend_palette(("#fa7de3", "#ffffff", "#6ef3ff"), n_colors=9, as_cmap=True, input='hex')

for ax in ax_cmap:
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])

# TRAIN MODELS AND PLOT
time_steps = 20
epochs = 30
num_predict = 30
start_time = time.time()

for i in range(1):

    print("Model", i, "| elapsed time:", "{:5.2f}".format((time.time() - start_time) / 60), "min")

    # Load a presaved model -- comment out figure 2
    # model = ODE_RNN(1, 4, 1, device_params, time_steps)
    # model.load_state_dict(torch.load("output/models/model.pt"))

    model = ODE_RNN(1, 4, 1, device_params, time_steps)
    torch.save(model.state_dict(), "output/models/model.pt")
    losses, val_losses = train.train(train_data, model, epochs)
    model.node_rnn.observe(True)
    # model.use_cb(True)

    output, times = train.test(
        test_start[0][0], test_start[0][1], num_predict, model)

    # model.use_cb(False)

    ax1.plot(torch.cat((x.view(-1)[cutoff + tw - 1].view(-1), times.view(-1)), axis=0),
             torch.cat(
                 (y.view(-1)[cutoff + tw - 1].view(-1), output.view(-1)), axis=0),
             'o',
             linewidth=1,
             color='aqua',
             markerfacecolor='none',
             )

    # Interpolation data
    H = model.node_rnn.observer.history[0].detach()
    t = model.node_rnn.observer.history[1].view(-1).detach()

    ax1.plot(t,
             model.decoder(torch.transpose(H, 0, 1)).view(-1).detach(),
             ':',
             linewidth=1,
             color='deeppink')

    ax2.plot(t,
             torch.linalg.norm(H, ord=2, dim=1).view(-1),
             ':',
             linewidth=1,
             color='deeppink')

    ax2.scatter(t[::(time_steps + 2)],
                torch.linalg.norm(H, ord=2, dim=1)[::(time_steps + 2)],
                linewidth=1,
                edgecolors='deeppink',
                facecolors='none')

    
    ax3.plot(list(range(epochs)),
             losses,
             linewidth=1, marker = 's',
             color='black')

    ax3.plot(list(range(epochs)),
            val_losses,
            linewidth=1, marker = 's',
            color='c')

    ax3.legend(('Training Loss', 'Validation Loss'), loc='right')

    unmapped_weights = torch.cat([tensor.reshape(-1).detach() for tensor in model.cb.tensors], axis=0)
    ax4.hist(unmapped_weights.numpy().reshape(-1), bins=20, color='pink')
    plt.setp(ax4, ylabel='# of weights in the bin')

    with torch.no_grad():
        left_mapped_weights = torch.cat([model.cb.W[m[0]:m[0]+m[2], m[1]:m[1]+m[3]:2].reshape(-1) for m in model.cb.mapped], axis=0).numpy().reshape(-1, 1)
        right_mapped_weights = torch.cat([model.cb.W[m[0]+1:m[0]+m[2]+1, m[1]+1:m[1]+m[3]+1:2].reshape(-1) for m in model.cb.mapped], axis=0).numpy().reshape(-1,1)
        ax5.hist(np.concatenate((left_mapped_weights, right_mapped_weights), axis=1), stacked=True, bins=100)

    weights = [model.cb.W[coord[0]:coord[0]+coord[2], coord[1]*2:coord[1]*2+coord[3]*2] for coord in model.cb.mapped] + [model.cb.W]
    vmax = max(torch.max(weight) for weight in weights)
    vmin = min(torch.min(weight) for weight in weights)

    with torch.no_grad():
        for i, weight in enumerate(weights):
            sns.heatmap(weight, vmax=vmax, vmin=vmin, cmap=cmap, square=True, cbar=False, ax=ax_cmap[i])

    print(model.cb.conductance_states.size())

ax1.plot(x.squeeze()[:cutoff+num_predict+tw], y.squeeze()[:cutoff +
                                                          num_predict+tw], linewidth=0.5, color='k', linestyle='dashed')
ax1.axvline(x=float(x.squeeze()[cutoff + tw - 1]), color='k')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

plt.setp(ax1, ylabel='real space')
ax1.legend(('predictions', 'interpolation', 'data'), loc='lower left')

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

plt.setp(ax2, xlabel='t (s)')
plt.setp(ax2, ylabel='norm of hidden state')

ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)

plt.setp(ax3, xlabel='Epoch')
plt.setp(ax3, ylabel='RMS Prediction Accuracy')

ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
plt.setp(ax4, xlabel='Unmapped Weights')

ax5.spines['right'].set_visible(False)
ax5.spines['top'].set_visible(False)
plt.setp(ax5, xlabel='Mapped Weights')
plt.setp(ax5, ylabel='# of weights in the bin')
ax5.legend(('Left Weights', 'Right Weights'), loc='lower right')

fig1.savefig('output/fig5/1.png', dpi=600, transparent=True)
fig3.savefig('output/fig5/2.png', dpi=600, transparent=True)
fig4.savefig('output/fig5/3.png', dpi=600, transparent=True)
fig5.savefig('output/fig5/4.png', dpi=600, transparent=True)

plt.show()
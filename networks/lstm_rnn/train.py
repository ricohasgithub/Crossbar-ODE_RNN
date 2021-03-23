
import torch

from .lstm_rnn_model import LSTM_RNN_Model

pi = 3.14159265359

# MAKE DATA
n_pts = 150
size = 1
tw = 25
cutoff = 50

x = torch.linspace(0, 24*pi, n_pts).view(1, -1)
y = torch.sin(x) / 2 + 0.5
data = [((y[:, i:i+tw].reshape(-1, size, 1), x[:, i:i+tw].reshape(-1, 1, 1)),
         (y[:, i+tw:i+tw+1].reshape(-1, size))) for i in range(y.size(1) - tw)]
train_data, test_start = data[:cutoff], data[cutoff]


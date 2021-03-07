
import torch
import torch.nn as nn

from utils.linear import Linear
from crossbar.crossbar import crossbar
from utils.observer import Observer
from networks.node_rnn import NODE_RNN

class ODE_RNN(nn.Module):

    def __init__(self, input_size, hidden_layer_size, output_size, device_params, time_steps):

        super(ODE_RNN, self).__init__()

        self.N = time_steps
        self.cb = crossbar(device_params)

        # Construct model and layers
        self.node_rnn = NODE_RNN(input_size, hidden_layer_size, self.cb, self.N)

        self.output_size = output_size
        self.linear_out = Linear(hidden_layer_size, output_size, self.cb)

    # Taking a sequence, this predicts the next N points, where
    def forward(self, data):
        return self.linear_out(self.node_rnn(*data))

    def remap(self):
        self.cb.clear()
        self.node_rnn.remap()
        self.linear.remap()
    
    def use_cb(self, state):
        self.node_rnn.use_cb(state)
        self.linear.use_cb(state)

import torch
import torch.nn as nn

from utils.linear import Linear
from crossbar.crossbar import crossbar
from utils.observer import Observer

from .node_rnn import NODE_RNN
from .node_rnn_decoder import NODE_RNN_Decoder

class ODE_RNN(nn.Module):

    def __init__(self, input_size, hidden_layer_size, output_size, device_params, time_steps):

        super(ODE_RNN, self).__init__()

        self.N = time_steps
        self.cb = crossbar(device_params)

        # Construct model and layers
        self.node_rnn = NODE_RNN(input_size, hidden_layer_size, self.cb, self.N)

        # Apply decoder
        self.output_size = output_size
        self.decoder = NODE_RNN_Decoder(hidden_layer_size, output_size, self.cb)

    # Taking a sequence, this predicts the next N points, where
    def forward(self, data):
        return self.decoder(self.node_rnn(*data))

    def remap(self):
        self.cb.clear()
        self.node_rnn.remap()
        self.decoder.remap()
    
    def use_cb(self, state):
        self.node_rnn.use_cb(state)
        self.decoder.use_cb(state)
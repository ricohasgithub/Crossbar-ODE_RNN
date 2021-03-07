
import torch
import torch.nn as nn

from utils.linear import Linear
from crossbar.crossbar import crossbar
from utils.observer import Observer
from networks.node_rnn import NODE_RNN

class NODE_RNN_Decoder(nn.Module):

    def __init__(self, hidden_layer_size, output_size, cb):

        super(NODE_RNN_Decoder, self).__init__()

        self.cb = cb

        # Construct model and layers
        self.output_size = output_size
        self.linear_out = Linear(hidden_layer_size, output_size, self.cb)

    # Taking a sequence, this predicts the next N points, where
    def forward(self, data):
        return self.linear_out(data)

    def remap(self):
        self.linear_out.remap()
    
    def use_cb(self, state):
        self.linear_out.use_cb(state)
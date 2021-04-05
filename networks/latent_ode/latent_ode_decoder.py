
import torch
import torch.nn as nn

from utils.linear import Linear
from crossbar.crossbar import crossbar
from utils.observer import Observer

class Latent_ODE_Decoder(nn.Module):

    def __init__(self, latent_dim, obs_dim, nhidden, cb):

        super(Latent_ODE_Decoder, self).__init__()

        self.cb = cb

        # Construct model and layers
        self.linear_out1 = Linear(latent_dim, nhidden, cb)
        self.linear_out2 = Linear(nhidden, obs_dim, cb)
        self.nonlinear = nn.ReLU()

    # Taking a sequence, this predicts the next N points, where
    def forward(self, data):
        out = self.linear_out1(data)
        out = self.nonlinear(out)
        out = self.linear_out2(out)
        return out

    def remap(self):
        self.linear_out.remap()
    
    def use_cb(self, state):
        self.linear_out.use_cb(state)
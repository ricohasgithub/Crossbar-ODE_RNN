
import torch
import torch.nn as nn

from utils.linear import Linear
from crossbar.crossbar import crossbar
from utils.observer import Observer

class Recognition_RNN(nn.Module):

    def __init__(self, latent_dims, obs_dims, nhidden, nbatch, cb):

        super(Recognition_RNN, self).__init__()

        self.cb = cb

        self.nhidden = nhidden
        self.nbatch = nbatch

        self.i2h = Linear(obs_dims + nhidden, nhidden, cb)
        self.h2o = Linear(nhidden, latent_dims * 2, cb)

    def forward(self, x, h):

        x = torch.transpose(x, 0, 1)
        # print("x:", x.size())
        # # h = torch.transpose(h, 0, 1)
        # print("h: ", h.size())

        combined = torch.cat((x, h), dim=0)
        print("combined: ", combined.size())

        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        # return torch.zeros(self.nbatch, self.nhidden)
        return torch.zeros(self.nhidden, self.nbatch)
    
    def remap(self):
        self.i2h.remap()
        self.h2o.remap()
    
    def use_cb(self, state):
        self.i2h.use_cb(state)
        self.h2o.use_cb(state)

import torch
import torch.nn as nn

from utils.linear import Linear
from crossbar.crossbar import crossbar
from utils.observer import Observer

class Latent_ODE_Net(nn.Module):

    def __init__(self, latent_dim, nhidden, cb):

        super(Latent_ODE_Net, self).__init__()
        
        self.cb = cb

        # Number of function evaluations
        self.nfe = 0

        self.elu = nn.ELU(inplace=True)
        self.linear1 = Linear(latent_dim, nhidden, cb)
        self.linear2 = Linear(nhidden, nhidden, cb)
        self.linear3 = Linear(nhidden, latent_dim, cb)

    def forward(self, t, x):

        self.nfe += 1

        # x = torch.transpose(x, 0, 1)

        out = self.linear1(x)
        out = self.elu(out)
        out = self.linear2(out)
        out = self.elu(out)
        out = self.linear3(out)

        # out = torch.transpose(out, 0, 1)
        
        return out

    def remap(self):
        self.linear1.remap()
        self.linear2.remap()
        self.linear3.remap()
    
    def use_cb(self, state):
        self.linear1.use_cb(state)
        self.linear2.use_cb(state)
        self.linear3.use_cb(state)
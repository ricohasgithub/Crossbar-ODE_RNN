
import torch
import torch.nn as nn

from networks.ode_rnn.ode_rnn import ODE_RNN as ODE_RNN

from networks.latent_ode.latent_ode_net import Latent_ODE_Net as ODE_Func
from networks.latent_ode.latent_ode_decoder import Latent_ODE_Decoder as Decoder

from utils.linear import Linear
from crossbar.crossbar import crossbar
from utils.observer import Observer

class Latent_ODE(nn.Module):

    def __init__(self, latent_dims, obs_dims, nhidden, input_size, hidden_layer_size, output_size, device_params, time_steps, cb):

        super(Latent_ODE, self).__init__()

        self.cb = cb

        self.ode_rnn = ODE_RNN(input_size, hidden_layer_size, output_size, device_params, time_steps, cb)

        self.nhidden = nhidden
        self.nbatch = nbatch

        self.i2h = Linear(obs_dims + nhidden, nhidden, cb)
        self.h2o = Linear(nhidden, latent_dims * 2, cb)

    def forward(self, x, h):
        pass

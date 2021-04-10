
import torch
import torch.nn as nn

from networks.ode_rnn.node_rnn import NODE_RNN as NODE_RNN

from networks.latent_ode.latent_ode_net import Latent_ODE_Net as ODE_Func
from networks.latent_ode.latent_ode_decoder import Latent_ODE_Decoder as Decoder

from utils.linear import Linear
from crossbar.crossbar import crossbar
from utils.observer import Observer

class Latent_ODE(nn.Module):

    def __init__(self, latent_dims, obs_dims, nhidden, input_size, hidden_layer_size, output_size, device_params, time_steps, cb):

        super(Latent_ODE, self).__init__()

        self.cb = cb

        self.ode_rnn = NODE_RNN(input_size, hidden_layer_size, output_size, cb, time_steps)
        self.decoder = Decoder(latent_dims, obs_dims, nhidden, cb)

        self.nhidden = nhidden
        self.nbatch = nbatch
        self.latent_dims = latent_dims
        self.obs_dims = obs_dims

        self.i2h = Linear(obs_dims + nhidden, nhidden, cb)
        self.h2o = Linear(nhidden, latent_dims * 2, cb)

    def forward(self, samp_trajs):

        h = initHidden()

        for t in reversed(range(samp_trajs.size(1))):
            obs = samp_trajs[:, t, :]
            print(obs.size())
            #out, h = forward(obs, h)
            self.ode_rnn(*obs, t)

        # x = torch.transpose(x, 0, 1)
        # combined = torch.cat((x, h), dim=0)

        # h = torch.tanh(self.i2h(combined))
        # out = self.h2o(h)

        out = torch.transpose(out, 0, 1)
        h = torch.transpose(h, 0, 1)

        qz0_mean, qz0_logvar = out[:, :self.latent_dims], out[:, self.latent_dims:]
        epsilon = torch.randn(qz0_mean.size())
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

        z0 = torch.transpose(z0, 0, 1)

        # Forward in time and solve ode for reconstructions
        pred_z = odeint(func, z0, samp_ts).permute(1, 0, 2)
        pred_x = self.decoder(pred_z)

        return pred_x

    def initHidden(self):
        return torch.zeros(self.nhidden, self.nbatch)

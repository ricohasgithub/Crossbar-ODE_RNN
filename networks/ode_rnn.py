
import torch
import torch.nn as nn

from utils.linear import Linear
from crossbar.crossbar import crossbar
from utils.observer import Observer
from networks.ode_net import ODE_Net

class ODE_RNN(nn.Module):

    def __init__(self, input_size, hidden_layer_size, output_size, device_params, time_steps):

        super(ODE_RNN, self).__init__()

        self.N = time_steps

        self.observer = Observer()
        self.cb = crossbar(device_params)

        # Construct model and layers
        self.input_size = input_size
        self.linear_in = Linear(input_size, hidden_layer_size, self.cb)

        self.hidden_layer_size = hidden_layer_size
        self.linear_hidden = Linear(hidden_layer_size, hidden_layer_size, self.cb)

        self.output_size = output_size
        self.linear_out = Linear(hidden_layer_size, output_size, self.cb)

        self.solve = ODE_Net(hidden_layer_size, self.N, self.cb, self.observer)
        self.nonlinear = nn.Tanh()

    # Taking a sequence, this predicts the next N points, where
    def forward(self, x, t):

        h_i = torch.zeros(self.hidden_layer_size, 1)

        for i, x_i in enumerate(x):
            if i == (len(x) - 1) and self.observer.on == True:
                self.solve.observer_flag = True
            
            h_i = self.solve(h_i, t[i-1] if i>0 else t[i], t[i])
            
            if i == (len(x) - 1):
                self.observer.append(h_i.view(1, -1), t[i])

            h_i = self.nonlinear(self.linear_in(x_i) + self.linear_hidden(h_i))

            if i == (len(x) - 1):
                self.observer.append(h_i.view(1, -1), t[i])

            self.solve.observer_flag = False

        return self.linear_out(h_i)

    def remap(self):
        self.cb.clear()
        self.linear_in.remap()
        self.linear_hidden.remap()
        self.solve.remap()
        self.linear_out.remap()
    
    def use_cb(self, state):
        self.linear_in.use_cb(state)
        self.linear_hidden.use_cb(state)
        self.solve.use_cb(state)
        self.linear_out.use_cb(state)

    def observe(self, state):
        self.observer.on = state
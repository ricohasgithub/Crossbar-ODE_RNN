
import torch
import torch.nn as nn

from utils.linear import Linear
from crossbar.crossbar import crossbar
from utils.observer import Observer

from .ode_net import *

class NODE_RNN(nn.Module):

    def __init__(self, input_size, hidden_layer_size, cb, time_steps):

        super(NODE_RNN, self).__init__()

        self.N = time_steps

        self.observer = Observer()
        self.cb = cb

        # Construct model and layers
        self.input_size = input_size
        self.linear_in = Linear(input_size, hidden_layer_size, self.cb)

        self.hidden_layer_size = hidden_layer_size
        self.linear_hidden = Linear(hidden_layer_size, hidden_layer_size, self.cb)

        # Append ODE Solver
        self.solve = Euler_Forward_ODE_Net(hidden_layer_size, self.N, self.cb, self.observer)
        self.nonlinear = nn.Tanh()

    # Taking a sequence, this predicts the next N points, where
    def forward(self, x, t):

        h_i = torch.zeros(self.hidden_layer_size, 1)

        for i, x_i in enumerate(x):
            if i == (len(x) - 1) and self.observer.on == True:
                self.solve.observer_flag = True

            # Solve step for Euler Forward
            h_i = self.solve(h_i, t[i-1] if i>0 else t[i], t[i])

            # Solve step for ODEint Net
            # print("tsize: ", t.size())
            # print("tsize: ", t.view(-1).size())
            # h_i = self.solve(t.view(-1), h_i)
            
            if i == (len(x) - 1):
                self.observer.append(h_i.view(1, -1), t[i])

            h_i = self.nonlinear(self.linear_in(x_i) + self.linear_hidden(h_i))

            if i == (len(x) - 1):
                self.observer.append(h_i.view(1, -1), t[i])

            self.solve.observer_flag = False

        return h_i

    def remap(self):
        self.linear_in.remap()
        self.linear_hidden.remap()
        self.solve.remap()
    
    def use_cb(self, state):
        self.linear_in.use_cb(state)
        self.linear_hidden.use_cb(state)
        self.solve.use_cb(state)

    def observe(self, state):
        self.observer.on = state
import torch
import torch.nn as nn

import utils.linear as Linear
import utils.observer
from crossbar import crossbar

class ODE_Net(nn.Module):

    """ Basic ODE Net implementing Euler's Method (to be expanded to higher order solvers)"""

    def __init__(self, hidden_layer_size, N, cb, observer):

        super(ODE_Net, self).__init__()

        # Set instance variables
        self.hidden_layer_size = hidden_layer_size
        self.cb = cb
        self.N = N

        self.linear = Linear.Linear(hidden_layer_size, hidden_layer_size, cb)
        self.nonlinear = nn.Tanh()

        self.observer_flag = False
        self.observer = observer
    
    def forward(self, x0, t0, t1):
        x, h = x0, (t1 - t0) / self.N
        for i in range(self.N):
            x = x + h * self.nonlinear(self.linear(x))
            if self.observer_flag:
                self.observer.append(x.view(1, -1), t0 + h*i)
        return x
    
    def remap(self):
        self.linear.remap()
       
    def use_cb(self, state):
        self.linear.use_cb(state)

    def observe(self, state):
        self.observer.on = state
import torch
import torch.nn as nn

from utils.linear import Linear
from crossbar.crossbar import crossbar
from utils.observer import Observer

from torchdiffeq import odeint

class Abstract_ODE_Net(nn.Module):

    def __init__(self, hidden_layer_size, N, cb, observer):

        super(Abstract_ODE_Net, self).__init__()

        # Set instance variables
        self.hidden_layer_size = hidden_layer_size
        self.cb = cb
        self.N = N

        self.linear = Linear(hidden_layer_size, hidden_layer_size, cb)
        self.nonlinear = nn.Tanh()

        self.observer_flag = False
        self.observer = observer
    
    def forward(self, x0, t0, t1):
        pass
    
    def remap(self):
        self.linear.remap()
       
    def use_cb(self, state):
        self.linear.use_cb(state)

    def observe(self, state):
        self.observer.on = state

class Euler_Forward_ODE_Net(Abstract_ODE_Net):

    """ Basic ODE Net Layer implementing Euler's Method (to be expanded to higher order solvers)"""

    def __init__(self, hidden_layer_size, N, cb, observer):
        super(Euler_Forward_ODE_Net, self).__init__(hidden_layer_size, N, cb, observer)
    
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

class ODE_Net(Abstract_ODE_Net):

    def __init__(self, hidden_layer_size, N, cb, observer):
        super(ODE_Net, self).__init__(hidden_layer_size, N, cb, observer)
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x0, t0, t1):
        self.integration_time = self.integration_time.type_as(x0)
        out = odeint(self.linear, x0, self.integration_time, 1e-3, 1e-3)
        return out[1]
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
        # N is a parallel to NFE (number of function evaluations)
        self.N = N

        self.linear = Linear(hidden_layer_size, hidden_layer_size, cb)
        self.nonlinear = nn.Tanh()

        self.observer_flag = False
        self.observer = observer
    
    def forward(self, t, x):
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

class ODE_Func(Abstract_ODE_Net):

    def __init__(self, hidden_layer_size, N, cb, observer):
        super(ODE_Func, self).__init__(hidden_layer_size, N, cb, observer)

    def forward(self, t, x):
        out = x + self.nonlinear(self.linear(x))
        return out

class ODE_Net(nn.Module):

    def __init__(self, hidden_layer_size, N, cb, observer):
        super(ODE_Net, self).__init__()
        self.ODE_Func = ODE_Func(hidden_layer_size, N, cb, observer)

    def forward(self, x0, t0, t1):
        # x is the parallel to y0
        x = x0
        t = torch.tensor([0, 1]).float()
        out = odeint(self.ODE_Func, x, t)
        return out[1]
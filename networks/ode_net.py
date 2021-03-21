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
        #self.linear2 = Linear(hidden_layer_size, hidden_layer_size, cb)
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
        # ODE Solve
        for i in range(self.N):
            x = x + h * self.nonlinear(self.linear(x))
            if self.observer_flag:
                self.observer.append(x.view(1, -1), t0 + h*i)
        return x

class ODE_Net(Abstract_ODE_Net):

    def __init__(self, hidden_layer_size, N, cb, observer):
        super(ODE_Net, self).__init__(hidden_layer_size, N, cb, observer)

    def forward(self, t, x):
        # x is the parallel to y0
        out = odeint(self.ODE_Func, x, t)
        # print("OUT1: ", out)
        # print(out.size())
        # print("T1: ", t)
        # print(t.size())
        # for i in t.size():
        #     if self.observer_flag:
        #         self.observer.apend(out[i].view(1, -1), t[i])
        # print("x: ", x.size())
        # print("t: ", t.size())
        # print("out: ", out.size())
        #return out
        return out[1]

    def ODE_Func(self, t, x):
        out = self.nonlinear(self.linear(x))
        # if self.observer_flag:
        #     print("OUT: ", out.size())
        #     print("T: ", t)
        #     self.observer.append(out.view(1, -1), t.reshape(-1))
        return out

import torch
import torch.nn as nn

class ODE_Func(nn.Module):
    
    def __init__(self, input_size, hidden_layer_size, output_size):

        super(ODE_Func, self).__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size

        # Replace linear with crossbar fed linear layers in custom crossbar package
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_layer_size),
            nn.Tanh(),
            nn.Linear(hidden_layer_size, output_size),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)

        
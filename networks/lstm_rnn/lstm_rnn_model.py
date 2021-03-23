
import torch
import torch.nn as nn

from utils.linear import Linear
from crossbar.crossbar import crossbar

class LSTM_RNN_Model(nn.Module):

    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, device_params):

        super(LSTM_RNN_Model, self).__init__()

        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

        self.cb = crossbar(device_params)

        self.lstm1 = nn.LSTMCell(input_layer_size, hidden_layer_size)
        self.lstm2 = nn.LSTMCell(hidden_layer_size, hidden_layer_size)
        self.linear = Linear(hidden_layer_size, output_layer_size, self.cb)
    
    def forward(self, input, future = 0):

        outputs = []

        h_t = torch.zeros(input.size(0), self.hidden_layer_size, dtype=torch.double)
        c_t = torch.zeros(input.size(0), self.hidden_layer_size, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), self.hidden_layer_size, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), self.hidden_layer_size, dtype=torch.double)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        outputs = torch.cat(outputs, dim=1)
        return outputs
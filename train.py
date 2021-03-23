
import torch
import torch.nn as nn
import torch.optim as optim

from networks.lstm_rnn.lstm_rnn_model import LSTM_RNN_Model

import numpy as np
import matplotlib as plt

pi = 3.14159265359

# DEVICE PARAMS for convenience.
device_params = {"Vdd": 1.8,
                 "r_wl": 20,
                 "r_bl": 20,
                 "m": 32,
                 "n": 32,
                 "r_on": 1e4,
                 "r_off": 1e5,
                 "dac_resolution": 4,
                 "adc_resolution": 14,
                 "bias_scheme": 1/3,
                 "tile_rows": 8,
                 "tile_cols": 8,
                 "r_cmos_line": 600,
                 "r_cmos_transistor": 20,
                 "p_stuck_on": 0.01,
                 "p_stuck_off": 0.01,
                 "method": "viability",
                 "viability": 0.2,
}

# MAKE DATA
n_pts = 150
size = 1
tw = 25
cutoff = 50

x = torch.linspace(0, 24*pi, n_pts).view(1, -1)
y = torch.sin(x) / 2 + 0.5
data = [((y[:, i:i+tw].reshape(-1, size, 1), x[:, i:i+tw].reshape(-1, 1, 1)),
         (y[:, i+tw:i+tw+1].reshape(-1, size))) for i in range(y.size(1) - tw)]
train_data, test_start = data[:cutoff], data[cutoff]

model = LSTM_RNN_Model(1, 20, 1, device_params)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 20

training_loss_history = []
validation_loss_history = []

# Split data into training and validation
example_data_length = len(train_data)
val_split_index = example_data_length - int(0.25 * example_data_length)

optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_function = nn.MSELoss()

for epoch in range(epochs):

    print("EPOCH: ", epoch)

    training_loss = []
    validation_loss = []

    for i, (example, label) in enumerate(train_data):
        
        # Validation
        if i > val_split_index:
            with torch.no_grad():
                prediction = model(example[0])
                loss = loss_function(prediction, label)
                validation_loss.append(loss)
        else:
            # Training
            optimizer.zero_grad()
            prediction = model(example[0])
            loss = loss_function(prediction, label)
            training_loss.append(loss)
            loss.backward()
            optimizer.step()

    # Append avereage loss over batch sample to history
    training_loss_history.append(sum(training_loss) / val_split_index)
    validation_loss_history.append(sum(validation_loss) / (example_data_length - val_split_index))
    
    training_loss = []
    validation_loss = []
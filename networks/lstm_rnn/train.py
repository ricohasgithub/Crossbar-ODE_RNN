
import torch
import torch.nn as nn
import torch.optim as optim

from networks.lstm_rnn.lstm_rnn import LSTM_RNN

import numpy as np
import matplotlib.pyplot as plt

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

x = np.arange(1,721,1)
y = np.sin(x*np.pi/180)

# structuring the data
X = []
Y = []
for i in range(0,710):
    list1 = []
    for j in range(i,i+10):
        list1.append(y[j])
    X.append(list1)
    Y.append(y[j+1])

#train test split
X = np.array(X)
Y = np.array(Y)
x_train = X[:360]
x_test = X[360:]
y_train = Y[:360]
y_test = Y[360:]

from torch.utils.data import Dataset

class timeseries(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.len = x.shape[0]

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
  
    def __len__(self):
        return self.len

dataset = timeseries(x_train,y_train)

from torch.utils.data import DataLoader
train_loader = DataLoader(dataset,shuffle=True,batch_size=256)

model = LSTM_RNN(1, 5, 1, device_params)
model.float()

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1200

training_loss_history = []
validation_loss_history = []

epoch_loss = []
validation_loss = []

#training loop
for i in range(epochs):

    index = 0

    for j, data in enumerate(train_loader):
        # y_pred = model(data[:][0].view(-1,10,1)).reshape(-1)
        # loss = loss_function(y_pred, data[:][1])
        # loss.backward()
        # optimizer.step()
        # epoch_loss.append(loss)
        if index == 0:
            y_pred = model(data[:][0].view(-1,10,1)).reshape(-1)
            loss = loss_function(y_pred, data[:][1])
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss)
            index = 1
        elif index == 1:
            with torch.no_grad():
                y_pred = model(data[:][0].view(-1,10,1)).reshape(-1)
                loss = loss_function(y_pred, data[:][1])
                validation_loss.append(loss)
                index = 0

    if i % 40 == 0 and i != 0:
        print(i,"th iteration : ", loss)
        training_loss_history.append(sum(epoch_loss)/40)
        validation_loss_history.append(sum(validation_loss)/40)
        epoch_loss = []
        validation_loss = []
    elif i % 40 == 0 and i == 0:
        print(i,"th iteration : ", loss)
        training_loss_history.append(sum(epoch_loss))
        validation_loss_history.append(sum(validation_loss))
        epoch_loss = []
        validation_loss = []

fig3, ax3 = plt.subplots()
fig3.suptitle('LSTM RNN Error')

ax3.plot(list(range(30)),
            training_loss_history,
            linewidth=1, marker = 's',
            color='black')

ax3.plot(list(range(30)),
        validation_loss_history,
        linewidth=1, marker = 's',
        color='c')

ax3.legend(('Training Loss', 'Validation Loss'), loc='right')

ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)

plt.setp(ax3, xlabel='Epoch')
plt.setp(ax3, ylabel='RMS Prediction Accuracy')

# test set actual vs predicted
# test_set = timeseries(x_test,y_test)
# test_pred = model(test_set[:][0].view(-1,10,1)).view(-1)
# plt.plot(test_pred.detach().numpy(),label='predicted')
# plt.plot(test_set[:][1].view(-1),label='original')
# plt.legend()

fig3.savefig('output/fig5/9.png', dpi=600, transparent=True)

plt.show()
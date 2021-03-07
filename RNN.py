
import torch
import torch.nn as nn
from torch.tensor import Tensor
from CustomDataset import TABLE
from torch.autograd import Variable

N_DIGITS = 6

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=50,
            hidden_size=128,
            num_layers=1, 
            batch_first=True,
        )
        
        self.out = nn.Linear(128, 36 * 6)

    def forward(self, x):
        # input x in the form of (batch_size, 1, height, width)
        # reformat it to (batch_size, time_steps=width, input_size=height)
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(100, 200, 50)
        
        r_out, (h_n, h_c) = self.rnn(x, None) 

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out

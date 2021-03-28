
import torch
import torch.nn as nn
from torch.tensor import Tensor
from CustomDataset import TABLE
from torch.autograd import Variable

N_DIGITS = 6

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(150, 128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=200*128//5,
                            hidden_size=256,
                            num_layers=2,
                            batch_first=True)

    def forward(self, x):
        # (128, 3, 200, 50) -> (128, 150, 200)
        x = x.reshape(-1, 150, 200).permute(0, 2, 1)
        x = x.reshape(-1, 150)
        fc1 = self.fc1(x)
        fc1 = fc1.reshape(-1, 5, 200*128//5)
        lstm, (h_n, h_c) = self.lstm(fc1, None)
        out = lstm[:, -1, :]

        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=256,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True)
        self.out = nn.Linear(6*128, 6 * 36)

    def forward(self, x):
        # (N, 256) -> (N, 6, 256) -> (N, 6, 128) -> (N, 6*128) -> (N, 6* 36) -> (N, 6, 36)
        x = x.reshape(-1, 1, 256) 
        x = x.expand(-1, 6, 256)
        lstm, (h_n, h_c) = self.lstm(x, None)
        y1 = lstm.reshape(-1, 6*128)
        out = self.out(y1)
        output = out.reshape(-1, 6, 36)
        return output


class RNN (nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)

        return decoder

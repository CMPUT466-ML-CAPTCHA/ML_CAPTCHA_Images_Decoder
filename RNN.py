


import torch
import torch.nn as nn
from torch.tensor import Tensor
from CustomDataset import TABLE
from torch.autograd import Variable

N_DIGITS = 6

class RNN(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(RNN, self).__init__()
        
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        
        self.hidden2in = nn.Linear(inputSize + hiddenSize, hiddenSize)
        self.hidden2out = nn.Linear(hiddenSize, outputSize)
        self.tanh = nn.Tanh()
        #self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x:Tensor, pre_state:Tensor):
        # input as [batchSize, seq_len, width, height]
        seqLen = x.shape[1]
        batchSize = x.shape[0]
        #x = torch.reshape(x, [batchSize, seqLen, self.inputSize])
        #x = x.permute(1, 0, 2)
        x = torch.reshape(x, [seqLen, batchSize, self.inputSize])
        a = Variable(torch.zeros(seqLen, batchSize, self.hiddenSize, device='cuda'))
        o = Variable(torch.zeros(seqLen, batchSize, self.outputSize, device='cuda'))
        if pre_state is None:
            pre_state = Variable(torch.zeros(batchSize, self.hiddenSize, device='cuda'))
        
        for t in range(seqLen):
            tmp = torch.cat((x[t], pre_state), 1)
            a[t] = self.hidden2in(tmp)
            hidden = self.tanh(a[t])
            
            pre_state = hidden
            o[t] = self.hidden2out(hidden)
        return o

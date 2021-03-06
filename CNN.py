import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=5, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer5 = nn.Linear(64*3*12, 512)
        self.out = nn.Linear(512, 36*6)

    def forward(self, x):
        # print(x.shape, '#0')  # torch.Size([64, 1, 50, 200]) #0
        x = self.layer1(x)
        # print(x.shape, '#1')  # torch.Size([64, 32, 25, 100]) #1
        x = self.layer2(x)
        # print(x.shape, '#2')  # torch.Size([64, 48, 12, 50]) #1
        x = self.layer3(x)
        # print(x.shape, '#3')  # torch.Size([64, 64, 6, 25]) #1
        x = self.layer4(x)
        # print(x.shape, '#4')  # torch.Size([64, 64, 3, 12]) #1
        x = x.view(-1, 64*3*12)
        # print(x.shape, '#5')
        x = self.layer5(x)
        # print(x.shape, '#6')
        output = self.out(x)
        return output

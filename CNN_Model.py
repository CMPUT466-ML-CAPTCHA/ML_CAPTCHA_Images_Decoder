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
        x = self.layer1(x)  # Input: torch.Size([64, 1, 50, 200])
        x = self.layer2(x)  # Input: torch.Size([64, 32, 25, 100])
        x = self.layer3(x)  # Input: torch.Size([64, 48, 12, 50])
        x = self.layer4(x)  # Input: torch.Size([64, 64, 6, 25])

        # Output: torch.Size([64, 64, 3, 12])
        x = x.view(-1, 64*3*12)
        x = self.layer5(x)
        output = self.out(x)
        # Output: torch.Size([64, 36*6])
        return output

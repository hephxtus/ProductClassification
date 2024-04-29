import math

import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    """ Models a simple Convolutional Neural Network"""

    def __init__(self, num_classes=10):
        """ initialize the network """
        super(Net, self).__init__()
        # 3 input image channel, 6 output channels,
        # 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Max pooling over a (2, 2) window
        # https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks
        self.pool = nn.MaxPool2d(2, 2) # kernel_size=(2x2), stride = the number of jumps a feature map must make per max pool operation
        self.conv2 = nn.Conv2d(6, 16, 5) # must be same in_channels as out_channels in conv1
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x:torch.Tensor):
        """
            the forward propagation algorithm
            https://towardsdatascience.com/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148
        """
        x: torch.Tensor = self.pool(F.relu(self.conv1(x)))
        x: torch.Tensor = self.pool(F.relu(self.conv2(x)))
        x: torch.Tensor = x.view(-1, 16 * 5 * 5) # or x.flatten
        x: torch.Tensor = F.relu(self.fc1(x))
        x: torch.Tensor = F.relu(self.fc2(x))
        x: torch.Tensor = self.fc3(x)
        return x


class Net2(nn.Module):
    def __init__(self, num_classes=10, image_dim=32):
        super(Net2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), # output size = ((input_size-kernel_size+2*padding)/stride) + 1
            # O = ((32 - 3 + 2 * 1)/1  + 1) = 32, depth (out channels) = 32, so dimensions = 32 * 32 * 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # halves output size
            # O = (32 - 2 + 2 * 1 )/2) = (16) * 16 * 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # increases depth to 64
            # 16 * 16 * 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # O = (16 - 2 + 2 * 1) / 2) = (8) * 8 * 64
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64 * int(image_dim/4)**2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

        for m in self.features.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.classifier.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
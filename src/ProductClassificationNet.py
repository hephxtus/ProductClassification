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



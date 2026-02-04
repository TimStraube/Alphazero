"""
description: Defines a residual block for use in neural networks.
secondary author: Tim Straube
licence: MIT
"""

import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    """
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(
            num_hidden, 
            num_hidden, 
            kernel_size=3, 
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(
            num_hidden, 
            num_hidden, 
            kernel_size=3, 
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
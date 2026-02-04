"""
description: Defines a residual network for use in neural networks.
secondary author: Tim Straube
licence: MIT
"""

import torch.nn as nn
from agents.alphazero.residualblock import ResidualBlock

class ResidualNetwork(nn.Module):
    """Residual network
    """
    def __init__(
        self, 
        game, 
        num_resBlocks, 
        num_hidden, 
        inputarrays, 
        device):
        
        super().__init__()
        self.device = device
        # Network definition
        self.startBlock = nn.Sequential(
            nn.Conv2d(
                inputarrays, 
                num_hidden, 
                kernel_size = 3, 
                padding = 1
            ),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        # ResidualBlocks
        self.backBone = nn.ModuleList([
            ResidualBlock(num_hidden) for i in range(num_resBlocks)
        ])
        # Policy head for stochastic policy prediction
        self.policyHead = nn.Sequential(
            nn.Conv2d(
                num_hidden, 
                6, 
                kernel_size = 3, 
                padding = 1
            ),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                6 * game.rows * game.columns, 
                game.actions
            )
        )
        # Value head for value prediction
        self.valueHead = nn.Sequential(
            nn.Conv2d(
                num_hidden, 
                3, 
                kernel_size = 3, 
                padding = 1
            ),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                3 * game.rows * game.columns, 
                1
            ),
            nn.Tanh()
        )
        self.to(device)
    def forward(self, x):
        """Feed-forward layer
        """
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value
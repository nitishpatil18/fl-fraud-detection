"""
net.py

simple feedforward neural network for fraud classification.
kept small intentionally: the point of this project is the FL
mechanism, not model complexity.
"""

import torch
import torch.nn as nn

INPUT_DIM = 30  # Time + V1-V28 + Amount


class FraudNet(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)

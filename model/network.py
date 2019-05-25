#!/usr/bin/env python
#! coding: utf-8

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.input_2d = nn.Linear(12, 32)
        self.fc1 = nn.Linear(32, 16)
        self.input_3d = nn.Linear(12, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x_2d, x_3d):
        x = F.relu(self.input_2d(x_2d))
        x = F.relu(self.fc1(x))
        y = F.relu(self.input_3d(x_3d))
        y = F.relu(self.fc2(y))
        z = torch.cat([x, y], dim=0)
        z = F.relu(self.fc3(z))
        return F.log_softmax(z, dim=0)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = Network().to(device)
    print(model)
    a = torch.rand(4, 3).reshape(-1).to(device)
    print(a)
    b = torch.rand(3, 4).reshape(-1).to(device)
    print(b)
    model.forward(a, b)

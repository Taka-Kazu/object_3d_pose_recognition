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
    def __init__(self, class_num):
        super(Network, self).__init__()
        self.input_onehot = nn.Linear(class_num, 16)
        self.fc1 = nn.Linear(16, 1)
        self.input_3d = nn.Linear(14, 64)
        self.fc2 = nn.Linear(64, 63)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.output = nn.Linear(64, 7)

    def forward(self, onehot, x_3d):
        x_3d_vec = x_3d[0:14]
        x_3d_center = x_3d[14:17]
        x = F.relu(self.input_onehot(onehot))
        x = F.relu(self.fc1(x))
        y = F.relu(self.input_3d(x_3d_vec))
        y = F.relu(self.fc2(y))
        z = torch.cat([x, y], dim=0)
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        z = self.output(z)
        z[0:3] = z[0:3] + x_3d_center
        return z

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = Network(3).to(device)
    print(model)
    a = torch.rand(3).to(device)
    print(a)
    b = torch.rand(17).to(device)
    print(b)
    print(model.forward(a, b))

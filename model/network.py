#!/usr/bin/env python
#! coding: utf-8

from __future__ import print_function
import argparse
import math as m
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Network(nn.Module):
    def __init__(self, class_num):
        super(Network, self).__init__()
        self.class_num = class_num
        # right
        self.min_x = -5.0
        self.max_x = 5.0
        self.dx = 0.1
        self.dim_x = int((self.max_x - self.min_x) / self.dx) + 1
        # under
        self.min_y = -5.0
        self.max_y = 5.0
        self.dy = 0.1
        self.dim_y = int((self.max_y - self.min_y) / self.dy) + 1
        # front
        self.min_z = -1.0
        self.max_z = 5.0
        self.dz = 0.1
        self.dim_z = int((self.max_z - self.min_z) / self.dz) + 1

        self.min_yaw = -m.pi
        self.max_yaw = m.pi
        self.dyaw = m.pi / 16.0
        self.dim_yaw = int((self.max_yaw - self.min_yaw) / self.dyaw) + 1
        self.min_h = 0.0
        self.max_h = 5.0
        self.dh = 0.2
        self.dim_h = int((self.max_h - self.min_h) / self.dh) + 1
        self.min_w = 0.0
        self.max_w = 10.0
        self.dw = 0.2
        self.dim_w = int((self.max_w - self.min_w) / self.dw) + 1
        self.min_l = 0.0
        self.max_l = 10.0
        self.dl = 0.2
        self.dim_l = int((self.max_l - self.min_l) / self.dl) + 1

        self.input_onehot = nn.Linear(self.class_num, 16)
        self.fc1 = nn.Linear(16, 1)
        self.input_3d = nn.Linear(14, 64)
        self.fc2 = nn.Linear(64, 63)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.prob_x = nn.Linear(64, self.dim_x)
        self.prob_y = nn.Linear(64, self.dim_y)
        self.prob_z = nn.Linear(64, self.dim_z)
        self.prob_yaw = nn.Linear(64, self.dim_yaw)
        self.prob_h = nn.Linear(64, self.dim_h)
        self.prob_w = nn.Linear(64, self.dim_w)
        self.prob_l = nn.Linear(64, self.dim_l)

    def forward(self, data):
        a_3d_vec = data[:, 0:14]
        a_3d_center = data[:, 14:17]

        onehot = torch.eye(self.class_num)[data[:, 17].long()]
        onehot = onehot.to(data.device)

        a = F.relu(self.input_onehot(onehot))
        a = F.relu(self.fc1(a))

        b = F.relu(self.input_3d(a_3d_vec))
        b = F.relu(self.fc2(b))

        c = torch.cat([a, b], dim=1)
        c = F.relu(self.fc3(c))
        c = F.relu(self.fc4(c))

        prob_x = F.softmax(self.prob_x(c), dim=1)
        prob_y = F.softmax(self.prob_y(c), dim=1)
        prob_z = F.softmax(self.prob_z(c), dim=1)
        prob_yaw = F.softmax(self.prob_yaw(c), dim=1)
        prob_h = F.softmax(self.prob_h(c), dim=1)
        prob_w = F.softmax(self.prob_w(c), dim=1)
        prob_l = F.softmax(self.prob_l(c), dim=1)

        return (prob_x, prob_y, prob_z, prob_yaw, prob_h, prob_w, prob_l)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = Network(3).to(device)
    print(model)
    b = torch.rand(2, 18).to(device)
    print(b)
    print('forward')
    c = model.forward(b)
    print(c)

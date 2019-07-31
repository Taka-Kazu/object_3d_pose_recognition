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
    def __init__(self, class_num, hidden_num=64):
        super(Network, self).__init__()
        self.class_num = class_num
        # right
        self.min_x = -5.0
        self.max_x = 5.0
        self.dx = 0.2
        self.dim_x = int((self.max_x - self.min_x) / self.dx) + 1
        # under
        self.min_y = -5.0
        self.max_y = 5.0
        self.dy = 0.2
        self.dim_y = int((self.max_y - self.min_y) / self.dy) + 1
        # front
        self.min_z = -5.0
        self.max_z = 5.0
        self.dz = 0.2
        self.dim_z = int((self.max_z - self.min_z) / self.dz) + 1

        self.min_yaw = -m.pi / 2.0
        self.max_yaw = m.pi / 2.0
        self.dyaw = m.pi / 16.0
        self.dim_yaw = int((self.max_yaw - self.min_yaw) / self.dyaw) + 1
        self.min_h = 0.0
        self.max_h = 5.0
        self.dh = 0.1
        self.dim_h = int((self.max_h - self.min_h) / self.dh) + 1
        self.min_w = 0.0
        self.max_w = 5.0
        self.dw = 0.1
        self.dim_w = int((self.max_w - self.min_w) / self.dw) + 1
        self.min_l = 0.0
        self.max_l = 10.0
        self.dl = 0.1
        self.dim_l = int((self.max_l - self.min_l) / self.dl) + 1

        num_onehot_hidden = 16
        self.num_3d_vector = 14
        self.input_onehot = nn.Linear(self.class_num, num_onehot_hidden)
        # self.input_onehot_bn = nn.BatchNorm1d(num_onehot_hidden, num_onehot_hidden)
        self.fc1 = nn.Linear(num_onehot_hidden, 1)
        self.bn1 = nn.BatchNorm1d(1, 1)
        self.input_3d = nn.Linear(self.num_3d_vector, hidden_num)
        # self.input_3d_bn = nn.BatchNorm1d(hidden_num, hidden_num)
        self.fc2 = nn.Linear(hidden_num, hidden_num-1)
        self.bn2 = nn.BatchNorm1d(hidden_num-1, hidden_num-1)
        self.fc3 = nn.Linear(hidden_num, hidden_num)
        self.bn3 = nn.BatchNorm1d(hidden_num, hidden_num)
        self.fc4 = nn.Linear(hidden_num, hidden_num)
        self.bn4 = nn.BatchNorm1d(hidden_num, hidden_num)
        self.prob_x = nn.Linear(hidden_num, self.dim_x)
        self.prob_y = nn.Linear(hidden_num, self.dim_y)
        self.prob_z = nn.Linear(hidden_num, self.dim_z)
        self.prob_yaw = nn.Linear(hidden_num, self.dim_yaw)
        self.prob_h = nn.Linear(hidden_num, self.dim_h)
        self.prob_w = nn.Linear(hidden_num, self.dim_w)
        self.prob_l = nn.Linear(hidden_num, self.dim_l)

    def forward(self, data):
        # print('forwarding data:\n', data)
        a_3d_vec = data[:, 0:self.num_3d_vector]
        a_3d_center = data[:, self.num_3d_vector:self.num_3d_vector+3]

        onehot = torch.eye(self.class_num)[data[:, 17].long()]
        onehot = onehot.to(data.device)

        a = self.input_onehot(onehot)
        # a = self.input_onehot_bn(a)
        a = F.relu(a)
        a = self.fc1(a)
        # a = self.bn1(a)
        a = F.relu(a)
        # print(a)

        b = self.input_3d(a_3d_vec)
        # b = self.input_3d_bn(b)
        b = F.relu(b)
        b = self.fc2(b)
        # b = self.bn2(b)
        b = F.relu(b)
        # print(b)

        c = torch.cat([a, b], dim=1)
        # print(c)
        c = self.fc3(c)
        # c = self.bn3(c)
        c = F.relu(c)
        # print(c)
        c = self.fc4(c)
        # c = self.bn4(c)
        c = F.relu(c)
        # print(c)

        prob_x = self.prob_x(c)
        prob_y = self.prob_y(c)
        prob_z = self.prob_z(c)
        prob_yaw = self.prob_yaw(c)
        prob_h = self.prob_h(c)
        prob_w = self.prob_w(c)
        prob_l = self.prob_l(c)

        return (prob_x, prob_y, prob_z, prob_yaw, prob_h, prob_w, prob_l)

    def convert_probability_to_prediction(self, prob_x, prob_y, prob_z, prob_yaw, prob_h, prob_w, prob_l):
        dev = prob_x.device
        cpu = torch.device('cpu')
        x = prob_x.detach().to(cpu).numpy().argmax(axis=1) * self.dx + self.min_x
        x = torch.Tensor(x).to(dev)
        y = prob_y.detach().to(cpu).numpy().argmax(axis=1) * self.dy + self.min_y
        y = torch.Tensor(y).to(dev)
        z = prob_z.detach().to(cpu).numpy().argmax(axis=1) * self.dz + self.min_z
        z = torch.Tensor(z).to(dev)
        yaw = prob_yaw.detach().to(cpu).numpy().argmax(axis=1) * self.dyaw + self.min_yaw
        yaw = torch.Tensor(yaw).to(dev)
        h = prob_h.detach().to(cpu).numpy().argmax(axis=1) * self.dh + self.min_h
        h = torch.Tensor(h).to(dev)
        w = prob_w.detach().to(cpu).numpy().argmax(axis=1) * self.dw + self.min_w
        w = torch.Tensor(w).to(dev)
        l = prob_l.detach().to(cpu).numpy().argmax(axis=1) * self.dl + self.min_l
        l = torch.Tensor(l).to(dev)
        return (x, y, z, yaw, h, w, l)

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

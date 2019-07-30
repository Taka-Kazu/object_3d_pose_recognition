#!/usr/bin/env python
#! coding: utf-8

from __future__ import print_function
import argparse
import os
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from network import Network
from dataset import ObjectDataset, ToTensor

def init_worker_seed(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def main():
    torch.manual_seed(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    kwargs = {'num_workers': 1, 'pin_memory': True, 'worker_init_fn': init_worker_seed}

    data_loader = torch.utils.data.DataLoader(
        ObjectDataset(os.path.dirname(os.path.abspath(__file__)) + '/../kitti', 'val', transform=transforms.Compose([
            ToTensor()
        ])),
        batch_size=1, shuffle=False, **kwargs)

    model_file = os.path.dirname(os.path.abspath(__file__)) + '/../models/' + 'object_recognition.pt'
    print('loading trained model from {}'.format(model_file))
    model = Network(3).to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    for m in model.parameters():
        print(m)

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(data_loader):
            data, label = data.to(device), label.to(device)
            print('=========== batch index ===========')
            print(batch_idx)
            print('=========== data ===========')
            print(data)
            print(data.device)
            print('=========== label ===========')
            print(label)
            print(label.device)
            print('=========== prediction ==========')
            prob_x, prob_y, prob_z, prob_yaw, prob_h, prob_w, prob_l = model(data)
            x, y, z, yaw, h, w, l = model.convert_probability_to_prediction(prob_x, prob_y, prob_z, prob_yaw, prob_h, prob_w, prob_l)
            print(x, y, z, yaw, h, w, l)
            print(x + data[:, 14], y + data[:, 15], z + data[:, 16], yaw, h, w, l)
            if(batch_idx == 3):
                break

if __name__ == '__main__':
    main()

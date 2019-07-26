#!/usr/bin/env python
#! coding: utf-8

from __future__ import print_function
import argparse
import os
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from network import Network
from dataset import ObjectDataset, ToTensor

log_directory = os.path.dirname(os.path.abspath(__file__)) + '/../logs/' + '{0:%Y%m%d%H%M%S}'.format(datetime.datetime.now())

training_step = 0
train_writer = SummaryWriter(log_dir=log_directory+'/train')
validation_writer = SummaryWriter(log_dir=log_directory+'/val')

def train(args, model, device, train_loader, optimizer, criterion, epoch):
    global training_step
    global train_writer

    model.train()

    train_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # print("shape: ", data.size())
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))
        # train_writer.add_scalar('loss', loss, training_step)
        training_step += 1
    train_loss /= len(train_loader)
    print('Train Epoch: {}\tAverage Loss: {:.6f}'.format(epoch, train_loss))
    train_writer.add_scalar('loss', train_loss, training_step)

def validation(args, model, device, test_loader, criterion):
    global training_step
    global validation_writer

    model.eval()

    test_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
    test_loss /= len(test_loader)
    print('\nValidation set: Average loss: {:.4f}\n'.format(test_loss))
    validation_writer.add_scalar('loss', test_loss, training_step)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(
        ObjectDataset(os.path.dirname(os.path.abspath(__file__)) + '/../kitti', 'train', transform=transforms.Compose([
            ToTensor()
        ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        ObjectDataset(os.path.dirname(os.path.abspath(__file__)) + '/../kitti', 'val', transform=transforms.Compose([
            ToTensor()
        ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model = Network(3).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print('output log file to', log_directory)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, criterion, epoch)
        validation(args, model, device, test_loader, criterion)

    if (args.save_model):
        torch.save(model.state_dict(),"object_recognition.pt")

    train_writer.close()
    validation_writer.close()

if __name__ == '__main__':
    main()

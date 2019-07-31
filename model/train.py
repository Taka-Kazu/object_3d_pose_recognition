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
from dataset import ObjectDataset, ToTensor, RandomHorizontalFlip

log_directory = os.path.dirname(os.path.abspath(__file__)) + '/../logs/' + '{0:%Y%m%d%H%M%S}'.format(datetime.datetime.now())

training_step = 0
train_writer = SummaryWriter(log_dir=log_directory+'/train')
validation_writer = SummaryWriter(log_dir=log_directory+'/val')

def init_worker_seed(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def train(args, model, device, train_loader, optimizer, criterion, epoch):
    global training_step
    global train_writer

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        prob_x, prob_y, prob_z, prob_yaw, prob_h, prob_w, prob_l = model(data)
        label_x = ((target[:, 0] - data[:, 14] - torch.tensor([model.min_x] * len(data), device=device)) / model.dx).long()
        loss_x = criterion['classification'](prob_x, label_x)
        label_y = ((target[:, 1] - data[:, 15] - torch.tensor([model.min_y] * len(data), device=device)) / model.dy).long()
        loss_y = criterion['classification'](prob_y, label_y)
        label_z = ((target[:, 2] - data[:, 16] - torch.tensor([model.min_z] * len(data), device=device)) / model.dz).long()
        loss_z = criterion['classification'](prob_z, label_z)
        label_yaw = ((target[:, 3] - torch.tensor([model.min_yaw] * len(data), device=device)) / model.dyaw).long()
        loss_yaw = criterion['classification'](prob_yaw, label_yaw)
        label_h = ((target[:, 4] - torch.tensor([model.min_h] * len(data), device=device)) / model.dh).long()
        loss_h = criterion['classification'](prob_h, label_h)
        label_w = ((target[:, 5] - torch.tensor([model.min_w] * len(data), device=device)) / model.dw).long()
        loss_w = criterion['classification'](prob_w, label_w)
        label_l = ((target[:, 6] - torch.tensor([model.min_l] * len(data), device=device)) / model.dl).long()
        loss_l = criterion['classification'](prob_l, label_l)
        _, _, _, yaw, h, w, l = model.convert_probability_to_prediction(prob_x, prob_y, prob_z, prob_yaw, prob_h, prob_w, prob_l)
        loss_yaw_reg = criterion['regression'](yaw, target[:, 3])
        loss_h_reg = criterion['regression'](h, target[:, 4])
        loss_w_reg = criterion['regression'](w, target[:, 5])
        loss_l_reg = criterion['regression'](l, target[:, 6])
        loss = loss_x + loss_y + loss_z + loss_yaw + loss_h + loss_w + loss_l
        loss += loss_yaw_reg + loss_h_reg + loss_w_reg + loss_l_reg
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        train_writer.add_scalar('loss', loss, training_step)
        train_writer.add_scalar('loss/x', loss_x, training_step)
        train_writer.add_scalar('loss/y', loss_y, training_step)
        train_writer.add_scalar('loss/z', loss_z, training_step)
        train_writer.add_scalar('loss/yaw', loss_yaw, training_step)
        train_writer.add_scalar('loss/h', loss_h, training_step)
        train_writer.add_scalar('loss/w', loss_w, training_step)
        train_writer.add_scalar('loss/l', loss_l, training_step)
        train_writer.add_scalar('loss/yaw_reg', loss_yaw_reg, training_step)
        train_writer.add_scalar('loss/h_reg', loss_h_reg, training_step)
        train_writer.add_scalar('loss/w_reg', loss_w_reg, training_step)
        train_writer.add_scalar('loss/l_reg', loss_l_reg, training_step)
        training_step += 1

def validation(args, model, device, test_loader, criterion):
    global training_step
    global validation_writer

    model.eval()

    test_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            prob_x, prob_y, prob_z, prob_yaw, prob_h, prob_w, prob_l = model(data)
            loss = 0
            label_x = ((target[:, 0] - data[:, 14] - torch.tensor([model.min_x] * len(data), device=device)) / model.dx).long()
            loss += criterion['classification'](prob_x, label_x)
            label_y = ((target[:, 1] - data[:, 15] - torch.tensor([model.min_y] * len(data), device=device)) / model.dy).long()
            loss += criterion['classification'](prob_y, label_y)
            label_z = ((target[:, 2] - data[:, 16] - torch.tensor([model.min_z] * len(data), device=device)) / model.dz).long()
            loss += criterion['classification'](prob_z, label_z)
            label_yaw = ((target[:, 3] - torch.tensor([model.min_yaw] * len(data), device=device)) / model.dyaw).long()
            loss += criterion['classification'](prob_yaw, label_yaw)
            label_h = ((target[:, 4] - torch.tensor([model.min_h] * len(data), device=device)) / model.dh).long()
            loss += criterion['classification'](prob_h, label_h)
            label_w = ((target[:, 5] - torch.tensor([model.min_w] * len(data), device=device)) / model.dw).long()
            loss += criterion['classification'](prob_w, label_w)
            label_l = ((target[:, 6] - torch.tensor([model.min_l] * len(data), device=device)) / model.dl).long()
            loss += criterion['classification'](prob_l, label_l)
            test_loss += loss
            _, _, _, yaw, h, w, l = model.convert_probability_to_prediction(prob_x, prob_y, prob_z, prob_yaw, prob_h, prob_w, prob_l)
            loss_yaw_reg = criterion['regression'](yaw, target[:, 3])
            loss_h_reg = criterion['regression'](h, target[:, 4])
            loss_w_reg = criterion['regression'](w, target[:, 5])
            loss_l_reg = criterion['regression'](l, target[:, 6])
            test_loss += loss_yaw_reg + loss_h_reg + loss_w_reg + loss_l_reg
    test_loss /= len(test_loader)
    print('\nValidation set: Average loss: {:.4f}\n'.format(test_loss))
    validation_writer.add_scalar('loss', test_loss, training_step)
    return test_loss

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='LR',
                        help='SGD momuntum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--random-batch', action='store_true', default=False,
                        help='if true, randomly sample batch')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True, 'worker_init_fn': init_worker_seed}

    train_loader = torch.utils.data.DataLoader(
        ObjectDataset(os.path.dirname(os.path.abspath(__file__)) + '/../kitti', 'train', transform=transforms.Compose([
            ToTensor(),
            RandomHorizontalFlip()
        ])),
        batch_size=args.batch_size, shuffle=args.random_batch, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        ObjectDataset(os.path.dirname(os.path.abspath(__file__)) + '/../kitti', 'val', transform=transforms.Compose([
            ToTensor()
        ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    model = Network(3).to(device)

    criterion = {'classification': F.nll_loss,
                 'regression': nn.SmoothL1Loss()}
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    print('output log file to', log_directory)

    min_val_loss = 1e6
    model_file = os.path.dirname(os.path.abspath(__file__)) + '/../models/' + 'object_recognition.pt'
    for epoch in range(1, args.epochs + 1):
        print('=== epoch {} ===\n'.format(epoch))
        train(args, model, device, train_loader, optimizer, criterion, epoch)
        val_loss = validation(args, model, device, test_loader, criterion)
        if min_val_loss > val_loss:
            print('min val loss is updated: {} -> {}\n'.format(min_val_loss, val_loss))
            min_val_loss = val_loss
            if args.save_model:
                torch.save(model.state_dict(), model_file)
                print('model is saved to {}\n'.format(model_file))
        else:
            print('current min val loss is {}\n'.format(min_val_loss))


    train_writer.close()
    validation_writer.close()

if __name__ == '__main__':
    main()

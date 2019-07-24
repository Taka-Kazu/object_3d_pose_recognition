#!/usr/bin/env python
#! coding: utf-8

from __future__ import print_function

import os
import _pickle as pickle
import torch
import torch.utils.data

class ObjectDataset(torch.utils.data.Dataset):
    def __init__(self, dir_name, prefix, transform=None):
        self.transform = transform
        self.data_num = 0
        self.data = []
        self.label = []

        # prepare data and label
        if prefix is 'train':
            file_name = dir_name + '/' + prefix + '.pickle'
            with open(file_name, 'rb') as fp:
                print('load ' + file_name)
                data = pickle.load(fp)
                print(data)
        elif prefix is 'test':
            pass
        else:
            print('invalid prefix')
            exit(0)

    def __len__(self):
        return self.data_num

    def __getitem(self, idx):
        output_data = self.data[idx]
        output_label = self.label[idx]

        if self.transform:
            output_data = self.transform(output_data)

        return output_data, output_label

if __name__ == '__main__':
    dataset = ObjectDataset(os.path.dirname(os.path.abspath(__file__)) + '/../kitti', 'train', transform=None)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)

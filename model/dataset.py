#!/usr/bin/env python
#! coding: utf-8

from __future__ import print_function

import os
import numpy as np
import _pickle as pickle
import torch
import torch.utils.data

'''
0 : ev1.x
1 : ev1.y
2 : ev1.z
3 : ev1
4 : ev2.x
5 : ev2.y
6 : ev2.z
7 : ev2
8 : ev3.x
9 : ev3.y
10: ev3.z
11: ev3
12: w
13: h
14: cp.x
15: cp.y
16: cp.z
17: class
18: file_index
19: gt.x
20: gt.y
21: gt.z
22: gt.yaw
23: gt.w
24: gt.h
25: gt.l
'''

class ObjectDataset(torch.utils.data.Dataset):
    def __init__(self, dir_name, prefix, transform=None):
        self.transform = transform
        self.data_num = 0
        data = []
        label = []
        self.classes = []
        with open(os.path.dirname(os.path.abspath(__file__)) + '/classes.txt', 'r') as fp:
            for line in fp:
                line = line.rstrip()
                self.classes.append(line)
        # print(self.classes)

        # prepare data and label
        if prefix is not 'train' and prefix is not 'test' and prefix is not 'val':
            print('invalid prefix')
            exit(0)

        file_name = dir_name + '/' + prefix + '.pickle'
        with open(file_name, 'rb') as fp:
            print('load ' + file_name)
            data = pickle.load(fp)
            classes = self.get_lower_case(data[:, 17])
            # string to id
            classes = [self.classes.index(class_) for class_ in classes]
            #print(classes)
            data[:, 17] = classes
            #print(data[0])
            #print(data[0, 0:18])
            #print(data[0, 19:26])
            self.data_num = len(data)
            data_ = self.get_float_data(data[:, 0:18])
            label_ = self.get_float_data(data[:, 19:26])
            self.sample = {'label': label_, 'data': data_}

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        output_sample = {'label': self.sample['label'][idx], 'data': self.sample['data'][idx]}

        if self.transform:
            output_sample = self.transform(output_sample)

        return (output_sample['data'], output_sample['label'])

    def get_float_data(self, dataset):
        _data = [[float(x) for x in data] for data in dataset]
        return _data

    def get_lower_case(self, string_dataset):
        _data = [string.lower() for string in string_dataset]
        return _data

class ToTensor(object):
    def __call__(self, sample):
        sample = {'label': torch.Tensor(sample['label']), 'data': torch.Tensor(sample['data'])}
        return sample

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        if np.random.rand() > 0.5:
            sample['label'][19 - 19] *= -1
            sample['data'][14] *= -1
            sample['data'][0] *= -1
            sample['data'][4] *= -1
            sample['data'][8] *= -1
        return sample

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from torchvision import transforms
    # dataset = ObjectDataset(os.path.dirname(os.path.abspath(__file__)) + '/../kitti', 'train', transform=None)
    # print(len(dataset))
    data_loader = torch.utils.data.DataLoader(
        ObjectDataset(os.path.dirname(os.path.abspath(__file__)) + '/../kitti', 'train', transform=transforms.Compose([
            ToTensor(),
            RandomHorizontalFlip()
        ])),
        batch_size=32, shuffle=False)
    print(len(data_loader))
    for batch_idx, (data, label) in enumerate(data_loader):
        print('=========== batch index ===========')
        print(batch_idx)
        print('=========== data ===========')
        print(data)
        data.to(device)
        print('=========== label ===========')
        print(label)
        label.to(device)

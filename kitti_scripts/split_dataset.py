#!/usr/bin/env python
#! coding: utf-8

from __future__ import print_function
import numpy as np
import os
import argparse

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(FILE_PATH)
DATASET_INDEX_PATH = ROOT_DIR + '/kitti/dataset_index/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='split_dataset')
    parser.add_argument('--train-ratio', type=float, default=0.5,
                        help='train data ratio in entire dataset (default: 0.5)')
    args = parser.parse_args()

    file_name = DATASET_INDEX_PATH + 'trainval.txt'

    train_ratio = args.train_ratio

    train_file_name = DATASET_INDEX_PATH + 'train_.txt'
    val_file_name = DATASET_INDEX_PATH + 'val_.txt'

    print('load data from {}'.format(file_name))

    indices = list()
    for line in open(file_name, 'r'):
        indices.append(line.rstrip())
    # print(indices)
    dataset_size = len(indices)
    print('dataset size is {}'.format(dataset_size))

    train_num = int(dataset_size * train_ratio)
    print('train dataset size is {}'.format(train_num))
    print('val dataset size is {}'.format(dataset_size - train_num))

    train_indices = np.random.choice(indices, train_num, replace=False)
    train_indices = np.sort(train_indices)
    np.savetxt(train_file_name, train_indices, fmt='%s')

    val_indices = np.setdiff1d(indices, train_indices)
    np.savetxt(val_file_name, val_indices, fmt='%s')

    print('\nsuccessfully generate files\n')

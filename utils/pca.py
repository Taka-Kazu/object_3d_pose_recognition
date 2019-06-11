#!/usr/bin/env python
#! coding utf-8

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import numpy as np

class PCA:
    def __init__(self, data):
        '''
        data: ndarray, n * 4
        '''
        data = data[:, 0:3]
        n = data.shape[0]
        m = data.shape[1]
        mean = np.mean(data, axis=0)
        covariance = np.cov(data.transpose())
        eigen_value, eigen_vector = np.linalg.eig(covariance)
        eigen_vector = eigen_vector.transpose()
        # sort
        contribution_rate = eigen_value / eigen_value.sum()
        descending_order_indices = np.argsort(eigen_value)[::-1]
        self.eigen_value = np.array([eigen_value[i].tolist() for i in descending_order_indices])
        self.eigen_vector = np.array([eigen_vector[i].tolist() for i in descending_order_indices])

    def get_eigen_vector(self):
        return self.eigen_vector

    def get_eigen_value(self):
        return self.eigen_value

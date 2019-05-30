#!/usr/bin/env python
#! coding utf-8

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import numpy as np
import scipy.spatial as ss

class EuclideanClustering:
    def __init__(self, data):
        '''
        data: ndarray, n * 3
        '''
        self.n = data.shape[0]
        self.m = data.shape[1]
        self.data = data

    def set_params(self, tolerance, min_cluster_size, max_cluster_size):
        self.tolorance = tolerance
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size

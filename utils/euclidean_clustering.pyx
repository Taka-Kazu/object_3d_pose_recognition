#!/usr/bin/env python
#! coding utf-8
# cython: language_level=3

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

cimport numpy as np
cimport cython
from libcpp cimport bool

import numpy as np
import scipy.spatial as ss
from pprint import pprint
import time

ctypedef np.float64_t DTYPE_t

cdef class EuclideanClustering:
    cdef double tolerance
    cdef int min_cluster_size
    cdef int max_cluster_size
    cdef int leaf_size

    def __init__(self):
        self.set_params()

    def set_params(self, tolerance_=0.5, min_cluster_size_=5, max_cluster_size_=2000, leaf_size_=100):
        self.tolerance = tolerance_
        self.min_cluster_size = min_cluster_size_
        self.max_cluster_size = max_cluster_size_
        self.leaf_size = leaf_size_

    cdef inline bool is_computed_index(self, int index, list indices_list):
        return index in indices_list

    cpdef calculate(self, data):
        return self.calculate_(data)

    cdef list calculate_(self, np.ndarray[DTYPE_t, ndim=2] data_):
        '''
        data: ndarray, n * 4
        '''
        cdef list data
        data = data_[:, 0:3].tolist()
        cdef int n = len(data)
        tree = ss.cKDTree(data, leafsize=self.leaf_size)
        cdef list queue = []
        cdef list cluster_indices_list = []
        cdef list computed_indices_list = []
        cdef int i, j, index
        cdef list indices, _indices
        cdef list p
        cdef bool computed_flag, max_flag

        for i in range(n):
            # computed_flag = self.is_computed_index(i, computed_indices_list)
            computed_flag = i in computed_indices_list
            if not computed_flag:
                queue.append(i)
                max_flag = False
                for j in queue:
                    p = data[j]
                    indices = tree.query_ball_point(p, self.tolerance)
                    _indices = list(set(indices) & set(computed_indices_list))
                    indices = list(set(indices) ^ set(_indices))
                    for index in indices:
                        if not (index in queue):
                            #print('index was added to queue')
                            queue.append(index)
                            if len(queue) == self.max_cluster_size:
                                max_flag = True
                                break
                        else:
                            #majority
                            #print('queue already has this index')
                            pass
                    if max_flag:
                        break
                cluster_indices_list.append(queue[:])
                computed_indices_list.extend(queue[:])
                del queue[:]
        # sort
        cluster_num = len(cluster_indices_list)
        while True:
            count = 0
            for i in range(cluster_num - 1):
                if len(cluster_indices_list[i]) < len(cluster_indices_list[i+1]):
                    buff = cluster_indices_list[i]
                    cluster_indices_list[i] = cluster_indices_list[i+1]
                    cluster_indices_list[i+1] = buff
                    count += 1
            if count == 0:
                break
        for indices in cluster_indices_list[:]:
            if len(indices) < self.min_cluster_size:
                cluster_indices_list.remove(indices)
        return cluster_indices_list

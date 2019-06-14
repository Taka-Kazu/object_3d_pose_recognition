#!/usr/bin/env python
#! coding utf-8

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import numpy as np
import scipy.spatial as ss
from pprint import pprint
import time

class EuclideanClustering:
    def __init__(self):
        self.set_params()

    def set_params(self, tolerance=0.5, min_cluster_size=5, max_cluster_size=2000, leaf_size=100):
        self.tolerance = tolerance
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.leaf_size = leaf_size

    def is_computed_index(self, index, indices_list):
        for indices in indices_list:
            if index in indices:
                return True
        return False

    def calculate(self, data):
        '''
        data: ndarray, n * 4
        '''
        self.data = data[:, 0:3].tolist()
        self.n = len(self.data)
        self.m = len(self.data[0])
        self.tree = ss.cKDTree(self.data, leafsize=self.leaf_size)
        queue = []
        queue_append = queue.append
        cluster_indices_list = []
        for i in range(self.n):
            computed_flag = self.is_computed_index(i, cluster_indices_list)
            if not computed_flag:
                queue_append(i)
                max_flag = False
                for j in queue:
                    p = self.data[j]
                    indices = self.tree.query_ball_point(p, self.tolerance)
                    for index in indices:
                        if not (index in queue) and not self.is_computed_index(index, cluster_indices_list):
                            #print('index was added to queue')
                            queue_append(index)
                            if len(queue) == self.max_cluster_size:
                                max_flag = True
                                break
                        else:
                            #print('queue already has this index')
                            pass
                    if max_flag:
                        break
                cluster_indices_list.append(queue[:])
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

if __name__=='__main__':
    ec = EuclideanClustering()
    pc = 3 * (np.random.rand(2000, 3) - 0.5)

    ec.set_params(1, 3, 6, 10)
    start = time.time()
    clusters = ec.calculate(pc)
    print(time.time() - start)
    #pprint(clusters)
    largest_cluster = pc[clusters[0]]
    #pprint(largest_cluster)

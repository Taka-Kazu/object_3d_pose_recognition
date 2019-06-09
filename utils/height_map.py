#!/usr/bin/env python
#! coding utf-8

# python implementation of https://github.com/jack-oquin/velodyne_height_map

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import numpy as np

class HeightMap:
    def __init__(self, cloud, cell_size=0.5, cell_num=320, height_threshold=0.25):
        '''
        cloud: ndarray, n * 4
        cell_size: double, [m/cell]
        cell_num: int, number of cell [cell]
        height_threshold: double, height difference threshold of points [m]
        '''
        cloud = cloud.tolist()

        self.obstacle_cloud = []
        self.ground_cloud = []

        min_ = [[0] * cell_num] * cell_num
        max_ = [[0] * cell_num] * cell_num
        init_ = [[False] * cell_num] * cell_num

        for pt in cloud:
            x = int((cell_num / 2) + pt[0] / cell_size)
            y = int((cell_num / 2) + pt[1] / cell_size)
            if x >= 0 and x < cell_num and y >= 0 and y < cell_num:
                if init_[x][y] == False:
                    min_[x][y] = pt[2]
                    max_[x][y] = pt[2]
                    init_[x][y] = True
                else:
                    min_[x][y] = min(min_[x][y], pt[2])
                    max_[x][y] = max(max_[x][y], pt[2])

        for pt in cloud:
            x = int((cell_num / 2) + pt[0] / cell_size)
            y = int((cell_num / 2) + pt[1] / cell_size)
            if x >= 0 and x < cell_num and y >= 0 and y < cell_num and init_[x][y]:
                if (max_[x][y] - min_[x][y]) > height_threshold:
                    self.obstacle_cloud.append(pt)
                else:
                    self.ground_cloud.append(pt)

    def get_obstacle_cloud(self):
        return np.array(self.obstacle_cloud)

    def get_ground_cloud(self):
        return np.array(self.ground_cloud)

if __name__=='__main__':
    from pprint import pprint
    pc = 3 * (np.random.rand(20, 4) - 0.5)
    pprint(pc)
    hm = HeightMap(pc)
    pprint(hm.get_obstacle_cloud())

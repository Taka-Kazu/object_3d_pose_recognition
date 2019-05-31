#!/usr/bin/env python
#! coding utf-8

import numpy as np

class BoundingBox3d:
    def __init__(self, data_size, data_xyz, yaw):
        self.h = data_size[0]
        self.w = data_size[1]
        self.l = data_size[2]
        self.position = np.array(data_xyz)
        self.yaw = yaw

#!/usr/bin/env python
#! coding utf-8

import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.bounding_box_2d import BoundingBox2d
from utils.bounding_box_3d import BoundingBox3d

class Object:
    def __init__(self, data):
        data = data.split(' ')
        # except label
        # string to float
        data[1:] = [float(x) for x in data[1:]]

        self.type = data[0]
        self.truncation = data[1] # 0 to 1

        class Visibility:
            def __init__(self):
                self.fully_visible = 0
                self.partly_visible = 1
                self.largely_occluded = 2
                self.unknown = 3

        self.visibility = data[2]
        self.object_direction_from_camera = data[3]

        self.bb2d = BoundingBox2d(data[4:8])

        self.bb3d = BoundingBox3d(data[8:11], data[11:14], data[14])

    def print_data(self):
        print('type: %s, truncation: %f, occlusion: %d, direction: %f' % (self.type, self.truncation, self.visibility, self.object_direction_from_camera))
        print('2D BBox (xmin, xmax, ymin, ymax): %f, %f, %f, %f' % (self.bb2d.xmin, self.bb2d.xmax, self.bb2d.ymin, self.bb2d.ymax))
        print('3D BBox h, w, l: %f, %f, %f' % (self.bb3d.h, self.bb3d.w, self.bb3d.l))
        print('3D BBox x, y, z: %f, %f, %f' % (self.bb3d.position[0], self.bb3d.position[1], self.bb3d.position[2]))
        print('3D BBox yaw: %f' % self.bb3d.yaw)

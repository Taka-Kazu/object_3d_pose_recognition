#!/usr/bin/env python
#! coding utf-8

import numpy as np

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

        class BoundingBox2d:
            def __init__(self, data):
                self.bb = np.array(data)
                self.xmin = self.bb[0]
                self.xmax = self.bb[1]
                self.ymin = self.bb[2]
                self.ymax = self.bb[3]

        self.bb2d = BoundingBox2d(data[4:8])

        class BoundingBox3d:
            def __init__(self, data_size, data_xyz, yaw_arround_camera_y):
                self.h = data_size[0]
                self.w = data_size[1]
                self.l = data_size[2]
                self.position = (data_xyz[0], data_xyz[1], data_xyz[2])
                self.yaw = yaw_arround_camera_y

        self.bb3d = BoundingBox3d(data[8:11], data[11:14], data[14])

    def print_data(self):
        print('type: %s, truncation: %f, occlusion: %d, direction: %f' % (self.type, self.truncation, self.visibility, self.object_direction_from_camera))
        print('2D BBox (xmin, xmax, ymin, ymax): %f, %f, %f, %f' % (self.bb2d.xmin, self.bb2d.xmax, self.bb2d.ymin, self.bb2d.ymax))
        print('3D BBox h, w, l: %f, %f, %f' % (self.bb3d.h, self.bb3d.w, self.bb3d.l))
        print('3D BBox x, y, z: %f, %f, %f' % (self.bb3d.position[0], self.bb3d.position[1], self.bb3d.position[2]))
        print('3D BBox yaw: %f' % self.bb3d.yaw)

#!/usr/bin/env python
#! coding utf-8

import numpy as np

class BoundingBox2d:
    def __init__(self, data):
        self.bb = np.array(data)
        self.xmin = self.bb[0]
        self.ymin = self.bb[1]
        self.xmax = self.bb[2]
        self.ymax = self.bb[3]
        #print('2D BBox (xmin, xmax, ymin, ymax): %f, %f, %f, %f' % (self.bb2d.xmin, self.bb2d.xmax, self.bb2d.ymin, self.bb2d.ymax))

    def get_hull(self):
        return np.array([[self.xmin, self.ymin],
                         [self.xmin, self.ymax],
                         [self.xmax, self.ymin],
                         [self.xmax, self.ymax]])

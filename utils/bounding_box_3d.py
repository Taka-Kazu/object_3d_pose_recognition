#!/usr/bin/env python
#! coding utf-8

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import numpy as np
from pprint import pprint

class BoundingBox3d:
    def __init__(self, data_size, data_xyz, yaw):
        self.h = data_size[0]
        self.w = data_size[1]
        self.l = data_size[2]
        self.position = np.array(data_xyz)
        self.yaw = yaw
        '''
           4-------6
          /|      /|
         / |     / |
        0-------2  |
        |  |    |  |
        |  5----|--7
        | /     | /
        |/      |/
        1-------3

             /z
            +--y
           x|

          x: length
          y: width
          z: height

          position is centroid of vertices
        '''
    def calculate_vertices(self):
        vertices = []
        for h in range(2):
            z = (h - 0.5) * self.h
            for w in range(2):
                y = (w - 0.5) * self.w
                for l in range(2):
                    x = (l - 0.5) * self.l
                    vertices.append([x, y, z])
        vertices = np.array(vertices)
        transform = np.array([[np.cos(self.yaw), -np.sin(self.yaw), 0, self.position[0]],
                              [np.sin(self.yaw),  np.cos(self.yaw), 0, self.position[1]],
                              [               0,                 0, 1, self.position[2]],
                              [               0,                 0, 0,                1]])
        vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        vertices = transform.dot(vertices.transpose())
        return vertices[0:3, :].transpose()

if __name__=="__main__":
    bb3d = BoundingBox3d([1, 2, 3], [0, 0, 0], 1)
    vertices = bb3d.calculate_vertices()
    pprint(vertices)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1.00)
    ax.axis('equal')
    ax.set_title('vertices')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

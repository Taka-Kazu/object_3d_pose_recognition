#!/usr/bin/env python
#! coding utf-8

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import cv2
import os
import numpy as np
#import mayavi.mlab as mlab
from pprint import pprint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

from kitti_object import Object
from calibration import Calibration
import loader

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(FILE_PATH)
KITTI_PATH = ROOT_DIR + '/dataset/kitti'
KITTI_TRAIN_PATH = KITTI_PATH + '/training'
KITTI_TEST_PATH = KITTI_PATH + '/test'

# for test
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file_index', help='default: 000000', default='000000')
    parser.add_argument('--show_image', action='store_true')
    parser.add_argument('--show_pointcloud', action='store_true')
    args = parser.parse_args()

    test_index = args.test_file_index

    image = loader.load_image(KITTI_TRAIN_PATH + '/image_2/' + test_index + '.png')
    print(image.shape)
    if args.show_image:
        window_name = 'test'
        cv2.namedWindow(window_name)
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

    pc = loader.load_pointcloud(KITTI_TRAIN_PATH + '/velodyne/' + test_index + '.bin')
    print(pc.shape)
    if args.show_pointcloud:
        #mlab.clf()
        #mlab.points3d(pc[:,0], pc[:,1], pc[:,2], pc[:,3])
        #raw_input()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        cloud = pc.transpose()
        ax.scatter3D(cloud[0], cloud[1], cloud[2], s=0.05)
        ax.set_title('pointcloud in ' + test_index)
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_zlim(-50, 50)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    objects = loader.load_label(KITTI_TRAIN_PATH + '/label_2/' + test_index + '.txt')
    for obj in objects:
        obj.print_data()

    calib = loader.load_camera_calibration(KITTI_TRAIN_PATH + '/calib/' + test_index + '.txt')
    c = Calibration(calib)
    p = objects[0].bb3d.position.reshape(-1, 3)
    print(p)
    p=c.project_3d_to_2d(p)
    print(p)
    p = np.hstack((p, objects[0].bb3d.position[2].reshape(-1, 1)))
    print(p)
    p=c.project_2d_to_3d(p)
    print(p)
    p=c.translate_p2_camera_to_velodyne(p)
    print(p)
    p=c.translate_velodyne_to_p2_image(p)
    print(p)

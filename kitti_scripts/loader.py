#!/usr/bin/env python
#! coding utf-8

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import cv2
import os
import numpy as np
import mayavi.mlab as mlab

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(FILE_PATH)
KITTI_PATH = ROOT_DIR + '/dataset/kitti'
KITTI_TRAIN_PATH = KITTI_PATH + '/training'
KITTI_TEST_PATH = KITTI_PATH + '/training'

def load_image(image_file_name):
    image = cv2.imread(image_file_name)
    return image

def load_pointcloud(pointcloud_file_name):
    pc = np.fromfile(pointcloud_file_name, dtype=np.float32)
    pc = pc.reshape((-1, 4))
    return pc

if __name__ == '__main__':
    image = load_image(KITTI_TRAIN_PATH + '/image_2/000000.png')
    window_name = 'test'
    #cv2.namedWindow(window_name)
    #cv2.imshow(window_name, image)
    #cv2.waitKey(0)
    #cv2.destroyWindow(window_name)

    pc = load_pointcloud(KITTI_TRAIN_PATH + '/velodyne/000000.bin')
    mlab.clf()
    mlab.points3d(pc[:,0], pc[:,1], pc[:,2], pc[:,3])

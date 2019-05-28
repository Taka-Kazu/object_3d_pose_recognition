#!/usr/bin/env python
#! coding utf-8

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import cv2
import os
import numpy as np
from pprint import pprint

from kitti_object import Object

def load_image(image_file_name):
    image = cv2.imread(image_file_name)
    return image

def load_pointcloud(pointcloud_file_name):
    # velodyne frame
    pc = np.fromfile(pointcloud_file_name, dtype=np.float32)
    pc = pc.reshape((-1, 4))
    return pc

def load_label(label_file_name):
    lines = []
    for line in open(label_file_name):
        # remove line feed character
        line = line.rstrip()
        lines.append(line.rstrip())
    objects = []
    for line in lines:
        obj = Object(line)
        objects.append(obj)
    return objects

def load_camera_calibration(calib_file_name):
    '''
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    '''
    data ={}
    with open(calib_file_name, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0:
                continue
            name, mat = line.split(':', 1)
            try:
                data[name] = np.array([float(x) for x in mat.split()])
            except ValueError:
                pass
    return data

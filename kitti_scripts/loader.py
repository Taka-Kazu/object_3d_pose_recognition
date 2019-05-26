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

from kitti_object import Object

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(FILE_PATH)
KITTI_PATH = ROOT_DIR + '/dataset/kitti'
KITTI_TRAIN_PATH = KITTI_PATH + '/training'
KITTI_TEST_PATH = KITTI_PATH + '/test'

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

class Calibration:
    def __init__(self, calib_file_name):
        calib = self.load_camera_calibration(calib_file_name)
        self.P2 = calib['P2'].reshape(3, 4)
        self.velodyne_to_camera = calib['Tr_velo_to_cam'].reshape(3, 4)
        self.velodyne_to_camera = np.vstack((self.velodyne_to_camera, np.zeros(4)))
        self.velodyne_to_camera[3, 3] = 1
        self.P0_R = np.array(calib['R0_rect'].reshape(3, 3))
        self.P0_R = np.vstack((self.P0_R, np.zeros((3))))
        self.P0_R = np.hstack((self.P0_R, np.zeros((4, 1))))
        self.P0_R[3, 3] = 1
        self.camera_params = {
            'c_u': self.P2[0, 2],
            'c_v': self.P2[1, 2],
            'f_u': self.P2[0, 0],
            'f_v': self.P2[1, 1],
            'b_x': self.P2[0, 3] / -self.P2[0, 0],
            'b_y': self.P2[1, 3] / -self.P2[1, 1]
        }

    def load_camera_calibration(self, calib_file_name):
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

    def translate_velodyne_to_p2_image(self, p):
        # velodyne xyz(n, 3) to p2 uv(n, 2)
        p = self.translate_velodyne_to_p0_camera(p)
        p = self.translate_p0_camera_to_p2_image(p)
        return p

    def translate_velodyne_to_p2_camera(self, p):
        # velodyne xyz(n, 3) to p2 xyz(n, 3)
        p = self.translate_velodyne_to_p0_camera(p)
        p = self.translate_p0_camera_to_p2_camera(p)
        return p

    def translate_velodyne_to_p0_camera(self, p):
        # velodyne xyz(n, 3) to p0 xyz(n, 3)
        p = self.add_dimension(p)
        p = p.transpose()
        p = self.velodyne_to_camera.dot(p)
        return p.transpose()[:, 0:3]

    def translate_p0_camera_to_p2_camera(self, p):
        # p0 xyz(n, 3) to p2 xyz(n, 3)
        p = self.add_dimension(p)
        p = p.transpose()
        p = self.P0_R.dot(p)
        return p.transpose()[:, 0:3]

    def translate_p0_camera_to_p2_image(self, p):
        # p0 xyz(n, 3) to p2 uv(n, 2)
        p = self.translate_p0_camera_to_p2_camera(p)
        p = self.project_3d_to_2d(p)
        return p

    def translate_p2_image_to_p0_camera(self, p):
        # p2 uvd(n, 3) to p0 xyz(n, 3)
        p = self.project_2d_to_3d(p)
        p = self.translate_p2_camera_to_p0_camera(p)
        return p

    def translate_p2_camera_to_p0_camera(self, p):
        # p2 xyz(n, 3) to p0 xyz(n, 3)
        p = self.add_dimension(p)
        p = p.transpose()
        p = np.linalg.inv(self.P0_R).dot(p)
        return p.transpose()[:, 0:3]

    def translate_p2_image_to_velodyne(self, p):
        # p2 uvd(n, 3) to velodyne xyz(n, 3)
        p = self.project_2d_to_3d(p)
        p = self.translate_p2_camera_to_velodyne(p)
        return p

    def translate_p2_camera_to_velodyne(self, p):
        # p2 xyz(n, 3) to velodyne xyz(n, 3)
        p = self.translate_p2_camera_to_p0_camera(p)
        p = self.translate_p0_camera_to_velodyne(p)
        return p

    def translate_p0_camera_to_velodyne(self, p):
        # p0 xyz(n, 3) to velodyne xyz(n, 3)
        p = self.add_dimension(p)
        p = p.transpose()
        p = np.linalg.inv(self.velodyne_to_camera).dot(p)
        return p.transpose()[:, 0:3]

    def project_3d_to_2d(self, p):
        # p2 xyz(n, 3) to p2 uv(n, 2)
        p = self.add_dimension(p)
        p = p.transpose()
        p = self.P2.dot(p)
        p = p.transpose()
        p[:, 0] /= p[:, 2]
        p[:, 1] /= p[:, 2]
        #p[:, 2] /= p[:, 2]
        p = p[:, 0:2]
        return p

    def project_2d_to_3d(self, p):
        # p2 uvd(n, 3) to p2 xyz(n, 3)
        # d is depth in p2_camera frame
        n = p.shape[0]
        u = p[:, 0]
        v = p[:, 1]
        d = p[:, 2]
        x = (u - self.camera_params['c_u']) * d / self.camera_params['f_u'] + self.camera_params['b_x']
        y = (v - self.camera_params['c_v']) * d / self.camera_params['f_v'] + self.camera_params['b_y']
        p = np.zeros((n, 3))
        p[:, 0] = x
        p[:, 1] = y
        p[:, 2] = d
        return p

    def add_dimension(self, p):
        # n*d to n*(d+1)
        n = p.shape[0]
        p = np.hstack((p, np.ones((n, 1))))
        return p

# for test
if __name__ == '__main__':
    test_index = '000000'
    image = load_image(KITTI_TRAIN_PATH + '/image_2/' + test_index + '.png')
    print(image.shape)
    #window_name = 'test'
    #cv2.namedWindow(window_name)
    #cv2.imshow(window_name, image)
    #cv2.waitKey(0)
    #cv2.destroyWindow(window_name)

    pc = load_pointcloud(KITTI_TRAIN_PATH + '/velodyne/' + test_index + '.bin')
    print(pc.shape)
    #mlab.clf()
    #mlab.points3d(pc[:,0], pc[:,1], pc[:,2], pc[:,3])

    objects = load_label(KITTI_TRAIN_PATH + '/label_2/' + test_index + '.txt')
    for obj in objects:
        obj.print_data()

    c = Calibration(KITTI_TRAIN_PATH + '/calib/' + test_index + '.txt')
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

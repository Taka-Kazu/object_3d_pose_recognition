#!/usr/bin/env python
#! coding utf-8

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import cv2
import os
import sys
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import _pickle as pickle
from tqdm import tqdm

from kitti_object import Object
from calibration import Calibration
import loader

# add project root dir to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.height_map import HeightMap
from utils.euclidean_clustering import EuclideanClustering
from utils.pca import PCA

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(FILE_PATH)
KITTI_PATH = ROOT_DIR + '/dataset/kitti'
KITTI_TRAIN_PATH = KITTI_PATH + '/training'
KITTI_TEST_PATH = KITTI_PATH + '/test'
DATASET_DIR = ROOT_DIR + '/kitti'

def plot_pointcloud(cloud, use_mayavi_flag=False):
    if not use_mayavi_flag:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        cloud = cloud.transpose()
        ax.scatter3D(cloud[0], cloud[1], cloud[2], s=0.05)
        ax.axis('equal')
        ax.set_title('pointcloud')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
    else:
        import mayavi.mlab as mlab
        mlab.clf()
        if cloud.shape[1] == 4:
            mlab.points3d(cloud[:,0], cloud[:,1], cloud[:,2], cloud[:,3], scale_factor=0.05)
        else:
            mlab.points3d(cloud[:,0], cloud[:,1], cloud[:,2], scale_factor=0.05)
        mlab.show()
        raw_input()

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0

def get_closest_point(cloud):
    closest_point = cloud[0]
    closest_distance = np.linalg.norm(closest_point)
    for pt in cloud:
        pt_distance = np.linalg.norm(pt)
        if pt_distance < closest_distance:
            closest_point = pt
            closest_distance = pt_distance
    return closest_point

def get_data_from_file_and_prepare(data_path, file_index, occlusion_list):
    '''
        file_index: string, like '000000'
    '''
    image = loader.load_image(data_path + '/image_2/' + file_index + '.png')
    print(image.shape)
    if args.show_image:
        window_name = 'test'
        cv2.namedWindow(window_name)
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

    pc = loader.load_pointcloud(data_path + '/velodyne/' + file_index + '.bin')
    print(pc.shape)
    if args.show_pointcloud:
        plot_pointcloud(pc, args.use_mayavi)

    objects = loader.load_label(data_path + '/label_2/' + file_index + '.txt')
    '''
    for obj in objects:
        obj.print_data()
    '''

    calib = loader.load_camera_calibration(data_path + '/calib/' + file_index + '.txt')
    c = Calibration(calib)
    '''
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
    '''

    print(pc.shape)
    hm = HeightMap(pc)
    pc = hm.get_obstacle_cloud()
    print('heightmap')
    print('%d points' % pc.shape[0])
    #print(hm.get_ground_cloud().shape)
    if args.show_pointcloud:
        plot_pointcloud(pc, args.use_mayavi)
    pc_without_intensity = pc[:, 0:3]
    projected_pointcloud = c.translate_velodyne_to_p2_image(pc_without_intensity)

    data = None
    for obj in objects:
        if obj.type == 'Pedestrian' or obj.type == 'Car':
            if obj.bb3d.position[2] < args.min_distance_limit:
                continue
            if obj.bb3d.position[2] > args.max_distance_limit:
                continue
            if not (obj.visibility in occlusion_list):
                continue
            obj.print_data()
            indices = in_hull(projected_pointcloud, obj.bb2d.get_hull())
            print('points in frustum')
            print('%d points' % projected_pointcloud[indices].shape[0])
            if projected_pointcloud[indices].shape[0] == 0:
                print("no point in 2d bounding box")
                continue
            object_pc = np.hstack((projected_pointcloud[indices], pc[indices, 0:1]))
            object_pc = c.translate_p2_image_to_p0_camera(object_pc)
            # restore intensity
            object_pc = np.hstack((object_pc, pc[indices, 2:3]))
            # delete z < 0 (behind camera)
            object_pc = np.delete(object_pc, np.where(object_pc[:, 2] < 0), axis=0)
            print('delete behind camera')
            print('%d points' % object_pc.shape[0])
            if args.show_pointcloud:
                plot_pointcloud(object_pc, args.use_mayavi)
            ec = EuclideanClustering()
            cluster_indices = ec.calculate(object_pc)
            if len(cluster_indices) == 0:
                print('no cluster!')
                continue
            object_pc = object_pc[cluster_indices[0]]
            print('cluster num: %d' % len(cluster_indices))
            print('final cluster')
            print('%d points' % object_pc.shape[0])
            if args.show_pointcloud:
                plot_pointcloud(object_pc, args.use_mayavi)
            pca = PCA(object_pc)
            eigen_value = pca.get_eigen_value()
            print(eigen_value)
            eigen_vector = pca.get_eigen_vector()
            print(eigen_vector)
            closest_point = get_closest_point(object_pc[:, 0:3])
            print('closest point: ', closest_point)
            bb_points = obj.bb2d.get_hull()
            d_list = np.full((bb_points.shape[0], 1), closest_point[2])
            # add depth
            bb_points = np.hstack((bb_points, d_list))
            bb_points = c.translate_p2_image_to_p0_camera(bb_points)
            #print(bb_points)
            w = bb_points[2, 0] - bb_points[0, 0]
            h = bb_points[1, 1] - bb_points[0, 1]
            print((w, h))
            # make data
            ## input data
            data_ = np.hstack((eigen_vector, eigen_value.reshape(-1, 1))).reshape(-1)
            data_ = np.hstack((data_, w, h))
            data_ = np.hstack((data_, closest_point))
            data_ = np.hstack((data_, obj.type))
            ## label
            data_ = np.hstack((data_, file_index))
            data_ = np.hstack((data_, obj.bb3d.position))
            data_ = np.hstack((data_, obj.bb3d.yaw))
            data_ = np.hstack((data_, obj.bb3d.size))
            object_pc_on_image = c.translate_p0_camera_to_p2_image(object_pc[:, 0:3])
            if args.show_image:
                img = image.copy()
                for pt in object_pc_on_image:
                    cv2.circle(img, (int(pt[0]), int(pt[1])), 1, (0, 0, 255), -1)
                cv2.rectangle(img, (obj.bb2d.xmin, obj.bb2d.ymin), (obj.bb2d.xmax, obj.bb2d.ymax), (255, 0, 0), 2)
                window_name = 'test'
                cv2.namedWindow(window_name)
                cv2.imshow(window_name, img)
                cv2.waitKey(0)
                cv2.destroyWindow(window_name)
            if data is not None:
                data = np.vstack((data, data_))
            else:
                data = data_
    if data is not None:
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
    else:
        return []
    return data.tolist()

def generate_data(prefix):
    index_file_name = os.path.join(DATASET_DIR, 'dataset_index', prefix + '.txt')
    print('load index from', index_file_name)
    data_index_list = ['%06d'%(int(line.rstrip())) for line in open(index_file_name)]
    data_path = None
    if prefix is 'train' or prefix is 'val':
        data_path = KITTI_TRAIN_PATH
    else:
        data_path = KITTI_TEST_PATH
    data = None
    for index in tqdm(data_index_list):
        print('===========================')
        print('data: ', index)
        data_ = get_data_from_file_and_prepare(data_path, index, occlusion_list)
        if data_ is None:
            continue
        if data is not None:
            # print(data.shape)
            print(np.array(data).shape)
            # data = np.vstack((data, data_))
            data.append(data_)
        else:
            data = data_

    output_file_name = os.path.join(DATASET_DIR, prefix + '.pickle')
    with open(output_file_name, 'wb') as fp:
        pickle.dump(data, fp)
    print('saved as ' + output_file_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_file_index', help='default: 000000', default='000000')
    parser.add_argument('--show_image', action='store_true')
    parser.add_argument('--show_pointcloud', action='store_true')
    parser.add_argument('--use_mayavi', action='store_true')
    parser.add_argument('--max_distance_limit', help='default: 25.0[m]', default=25.0)
    parser.add_argument('--min_distance_limit', help='default: 1.0[m]', default=1.0)
    parser.add_argument('--occlusion_list', help='default: 0,1,2', default='0,1,2')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--train', action='store_true', help='make train data')
    parser.add_argument('--val', action='store_true', help='make validation data')
    parser.add_argument('--test', action='store_true', help='make test data')
    args = parser.parse_args()

    if args.demo:
        test_index = args.demo_file_index
        occlusion_list = [int(x) for x in args.occlusion_list.split(',')]
        print(occlusion_list)

        data = get_data_from_file_and_prepare(KITTI_TRAIN_PATH, test_index, occlusion_list)

        print(data)
        output_file_name = DATASET_DIR + '/' + test_index + '.pickle'
        with open(output_file_name, 'wb') as fp:
            pickle.dump(data, fp)
        print('saved as ' + output_file_name)
        with open(output_file_name, 'rb') as fp:
            print('load ' + output_file_name)
            data = pickle.load(fp)
            print(data)
    else:
        occlusion_list = [int(x) for x in args.occlusion_list.split(',')]
        print('occlusion: ', occlusion_list)
        if args.train:
            print('=== generate train data ===')
            generate_data('train')
        if args.val:
            print('=== generate validation data ===')
            generate_data('val')
        if args.test:
            print('=== generate test data ===')
            generate_data('test')

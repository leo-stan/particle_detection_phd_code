#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename: visualise_dataset.py
Author: Leo Stanislas
Date Created:
Description: Visualise created datasets in ROS Rviz
"""

import numpy as np
import os.path as osp
import rospy
from cv_bridge import CvBridge
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2 as Pc2
import sensor_msgs.point_cloud2 as p_c2
from sensor_msgs.msg import PointField as Pf
from sensor_msgs.msg import Image
import sys
sys.path.append('../')
from particle_detection.src.config import cfg
from data_prep.ros_vis import RosVisualiser
import yaml
import PIL.Image
import cv2

data_dir = '/home/leo/phd/particle_detection/src/particle_detection/data'

# Load data
dataset = 'smoke_car_back_far'
img_rgb = True
img_label = False
overlay_rgb_labels = False
img_depth = False
lidar = False
lidar_refined = False
multisense = True
multisense_label = False
multisense_label_refined = False
multisense_sampled = False

# Load dataset parameters
with open(osp.join(data_dir, dataset, 'config.yaml'), 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

if lidar:
    lidar_name = 'pcl_labeled_spaces_converted_sync_cropped'
    if lidar_refined:
        lidar_name += '_refined'
    lidar_name += '.npy'
    pcls = np.load(osp.join(data_dir, dataset, lidar_name))
if img_rgb:
    imgs = np.load(osp.join(data_dir, dataset, 'images_sync.npy'))
if img_label:
    imgs_label = np.load(osp.join(data_dir, dataset, 'images_sync_label.npy'))
if img_depth:
    imgs_depth = np.load(osp.join(data_dir, dataset, 'images_sync_depth.npy'))
if multisense:

    multi_name = 'multi_pcl_sync'
    if multisense_sampled:
        multi_name += '_sampled'
    if multisense_label or multisense_label_refined:
        multi_name += '_labelled'
    multi_name+= '_projected'
    if multisense_label_refined:
        multi_name += '_refined'
    multi_name+= '.npy'
    multi_pcls = np.load(osp.join(data_dir, dataset, multi_name))

rospy.init_node('visualise_labeled_lidar_pcl', anonymous=True)
bridge = CvBridge()

header = Header()
header.frame_id = 'velodyne'
img_rgb_pub = rospy.Publisher("multisenseS21/left/image_color", Image, queue_size=1)
img_label_pub = rospy.Publisher("multisenseS21/left/image_label", Image, queue_size=1)
img_depth_pub = rospy.Publisher("multisenseS21/left/image_depth", Image, queue_size=1)
lidar_pub = rospy.Publisher("velodyne_points_labeled", Pc2, queue_size=1)
multi_pub = rospy.Publisher("multisenseS21/image_points2_color", Pc2, queue_size=1)

p_x = Pf('x', 0, 7, 1)
p_y = Pf('y', 4, 7, 1)
p_z = Pf('z', 8, 7, 1)
p_int = Pf('intensity', 12, 2, 1)
p_ring = Pf('echo', 13, 2, 1)
p_label = Pf('gt_label', 14, 2, 1)
p_color = Pf('rgb', 12, 7, 1)
fields = [p_x, p_y, p_z,p_int,p_ring,p_label]

if multisense_label:
    multi_fields = [p_x, p_y, p_z, p_color, Pf('gt_label', 16, 2, 1)]
else:
    multi_fields = [p_x, p_y, p_z, p_color]
ros_vis = RosVisualiser(max_id=config['count_sync'], rate=20, verbose=True)

while ros_vis.state is not 'exit':

    id = ros_vis.update_id()
    header.stamp = rospy.Time.now()
    if lidar:
        lidar_pub.publish(p_c2.create_cloud(header, fields, pcls[id]))
    if img_rgb:
        img_rgb_pub.publish(bridge.cv2_to_imgmsg(imgs[id], encoding="passthrough"))
    if img_label:
        if overlay_rgb_labels and img_rgb:
            # Label is RED
            # lbl = np.expand_dims(imgs_label[id], axis=2)
            lbl = np.tile((imgs_label[id]>85).astype(np.uint8), (1, 1, 3)) * np.array([0, 0, 255], dtype=np.uint8)

            alpha = 0.7
            proj_lbl = lbl.copy()

            cv2.addWeighted(imgs[id].astype(np.uint8), alpha, proj_lbl, 1 - alpha, 0, proj_lbl)
            # proj_lbl = proj_lbl[:, :, ::-1]
            img_label_pub.publish(bridge.cv2_to_imgmsg(proj_lbl, encoding="passthrough"))
        else:
            img_label_pub.publish(bridge.cv2_to_imgmsg(imgs_label[id], encoding="passthrough"))
    if img_depth:
        img_depth_pub.publish(bridge.cv2_to_imgmsg((imgs_depth[id] * 255.0/imgs_depth[id].max()).astype(np.uint8) , encoding="passthrough"))
    if multisense:
        multi_pub.publish(p_c2.create_cloud(header, multi_fields, multi_pcls[id]))
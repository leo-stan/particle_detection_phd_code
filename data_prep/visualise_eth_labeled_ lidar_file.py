#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename: visualise_eth_labeled_lidar_file.py
Author: Leo Stanislas
Date Created:
Description: Visualise lidar files labelled by Julian in ROS Rviz
"""

import cv2
import numpy as np
import copy
import time
import os.path as osp
import os
import rospy
from cv_bridge import CvBridge
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2 as Pc2
import sensor_msgs.point_cloud2 as p_c2
from sensor_msgs.msg import PointField as Pf
from sensor_msgs.msg import Image
import pdb
import sys
sys.path.append('../')
from particle_detection.src.config import cfg
from image_classifier.config import cfg as img_cfg

# Load data

filename = '2-dust'

print('loading dataset...')
# New dataset
pcl_new = np.load(osp.join(img_cfg.DATASETS_DIR,filename,'pcl_labeled_spaces_converted.npy'))

# Julian dataset
pcl_julian = np.load(osp.join(cfg.RAW_DATA_DIR,filename+'_converted.npy'))



rospy.init_node('visualise_labeled_lidar_pcl', anonymous=True)
bridge = CvBridge()

header = Header()
header.frame_id = 'velodyne'

lidar_pub_new = rospy.Publisher("velodyne_points_labeled", Pc2, queue_size=1)
lidar_pub_julian = rospy.Publisher("velodyne_points_labeled2", Pc2, queue_size=1)# Declare publisher

p_x = Pf('x', 0, 7, 1)
p_y = Pf('y', 4, 7, 1)
p_z = Pf('z', 8, 7, 1)
p_int = Pf('intensity', 12, 2, 1)
p_ring = Pf('echo', 13, 2, 1)
p_label = Pf('gt_label', 14, 2, 1)
fields = [p_x, p_y, p_z,p_int,p_ring,p_label]

rate = rospy.Rate(1)

# pcl_array = [p.reshape((2172,32,4)) for p in pcl]

print('publishing...')

for i in range(len(pcl_julian)):

    header.stamp = rospy.Time.now()
    lidar_pub_julian.publish(p_c2.create_cloud(header, fields, pcl_julian[len(pcl_julian)-1-i]))
    lidar_pub_new.publish(p_c2.create_cloud(header, fields, pcl_new[len(pcl_new)-1 -i]))
    rate.sleep()
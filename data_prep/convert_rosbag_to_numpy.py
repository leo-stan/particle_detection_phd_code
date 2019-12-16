#!/usr/bin/env python

# This script extract LiDAR, Stereo pcl and images from rosbag with time synchronisation
# sync_dataset.py must be run to remove empty scans added for synchronisation

import sys
sys.path.append('../')

import rosbag
import rospy
import sensor_msgs.point_cloud2 as p_c2
import numpy as np
import os.path as osp
import os
from cv_bridge import CvBridge
from particle_detection.src.config import cfg
import yaml
import argparse

field_names = [
        'x',
        'y',
        'z',
        'intensity',
        'ring']

multi_field_names = [
    'x',
    'y',
    'z',
    'rgb',
]

# List of bag files to open
# rosbags = [
    # '2018-08-31-14-06-17.bag',  # 1-dust.bag
    # '2018-08-31-14-07-39.bag',  # 2-dust.bag
    # '2018-08-31-14-15-25.bag',  # 3-dust.bag
    # '2018-08-31-14-18-44.bag',  # 4-dust.bag
    # '2018-08-31-14-20-56.bag',  # 5-dust.bag
    # '2018-08-31-14-33-26.bag',  # 6-dust.bag
    # '2018-08-31-14-39-56.bag',  # 7-dust.bag
    # '2018-08-31-14-44-39.bag',  # 8-dust.bag
    # '2018-08-31-12-12-25.bag', #  9-smoke.bag
    # '2018-08-31-12-16-11.bag', #  10-smoke.bag
    # '2018-08-31-12-19-44.bag', #  11-smoke.bag
    # '2018-08-31-12-21-53.bag', #  12-smoke.bag
    # '2018-08-31-12-24-10.bag', #  13-smoke.bag
    # '2018-08-31-12-26-55.bag', #  14-smoke.bag
    # '2018-08-31-12-29-18.bag', #  15-smoke.bag
    # '2018-08-31-12-31-35.bag', #  16-smoke.bag
    # '2018-08-31-12-35-24.bag', #  17-smoke.bag
    # '2018-08-31-12-38-14.bag', #  18.smoke.bag
    # '2018-08-31-12-40-31.bag', #  19-smoke.bag
    # '2018-04-10-12-32-35.bag', # smoke-bush

# ]

datasets = [
    # '1-dust',
    # '2-dust',
    # '3-dust',
    # '4-dust',
    # '5-dust',
    # '6-dust',
    # '7-dust',
    # '8-dust',
    # '9-smoke',
    # '10-smoke',
    # '11-smoke',
    # '12-smoke',
    # '13-smoke',
    # '14-smoke',
    # '15-smoke',
    # '16-smoke',
    # '17-smoke',
    # '18-smoke',
    # '19-smoke',
    'smoke_bush',
    # 'smoke_car',
    # 'smoke_car_back',
    # 'smoke_car_back_far',
]


bridge = CvBridge()

# Go through list of bag files
def convert_rosbag(dataset):

    print('Extracting %s...' % dataset)
    bag_data = rosbag.Bag(osp.join(cfg.ROSBAGS_DIR, dataset+'.bag'))  # load bag from string name
    msgs = bag_data.read_messages()  # read messages in bag
    pcl = []
    img = []
    multi_pcl = []
    last_img = np.array([])
    last_last_img = np.array([])
    last_pcl_first_echo = np.array([])
    last_pcl_second_echo = np.array([])
    last_multi_pcl = np.array([])
    vel_saved = False
    t_last_img = rospy.Time()
    t_last_pcl_second_echo = rospy.Time()
    t_last_multi_pcl = rospy.Time()
    c_vel_last = 0
    c_vel_first = 0
    c_img = 0
    c_saved = 0
    c_multi_pcl = 0
    t_diff_last_pcl = rospy.Time(0)
    t_diff_last_img = rospy.Time(0)

    # Go through each message in the bag
    for topic, msg, t in msgs:

        # Check the topic that the msg came from
        if topic == '/velodyne_points_dual':
            c_vel_first += 1
            # If last velodyne scan hasnt been saved then save it before overwriting
            if not vel_saved and last_pcl_first_echo.any():
                assert last_echo_seq == first_echo_seq
                pcl.append(np.concatenate((last_pcl_first_echo, last_pcl_second_echo), axis=1))
                # Save last img
                img.append(last_img)
                # append multi_pcl
                multi_pcl.append(last_multi_pcl)
                last_pcl_second_echo = np.array([])
                c_saved += 1

            last_pcl_first_echo = np.asarray(list(p_c2.read_points(msg, field_names=field_names, skip_nans=True)), dtype=np.float32)
            first_echo_seq = msg.header.seq
            vel_saved = False


        elif topic == '/velodyne_points':
            c_vel_last+=1
            # If last velodyne scan hasnt been saved then save it before overwriting
            if not vel_saved and last_pcl_second_echo.any():
                assert last_echo_seq == first_echo_seq
                pcl.append(np.concatenate((last_pcl_first_echo, last_pcl_second_echo), axis=1))

                #Save last img
                img.append(last_img)

                # append multi_pcl
                multi_pcl.append(last_multi_pcl)
                last_pcl_first_echo = np.array([])
                c_saved+=1

            last_pcl_second_echo = np.asarray(
                    list(p_c2.read_points(msg, field_names=field_names, skip_nans=True)), dtype=np.float32)
            t_last_pcl_second_echo = t
            last_echo_seq = msg.header.seq
            vel_saved = False

        elif topic == '/multisenseS21/left/image_color':
            c_img+=1

            # Update last img to new img
            last_last_img = last_img
            last_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            t_last_img = t
            # print('diff to last pcl %d' % (t_last_img - t_last_multi_pcl).nsecs)
            t_diff_last_pcl = t_last_img - t_last_multi_pcl
        elif topic == '/multisenseS21/image_points2_color':
            c_multi_pcl += 1

            if not vel_saved and last_pcl_first_echo.any() and last_pcl_second_echo.any():
                # if last_img.any():
                    # if last pcl is closer in time to last img then save pcl with last img, otherwise save pcl with new img
                assert last_echo_seq == first_echo_seq
                if abs(t_last_pcl_second_echo-t_last_multi_pcl) < abs(t_last_pcl_second_echo-t):
                    # save last img
                    pcl.append(np.concatenate((last_pcl_first_echo, last_pcl_second_echo), axis=1))
                    img.append(last_last_img)
                    multi_pcl.append(last_multi_pcl)
                else:
                    pcl.append(np.concatenate((last_pcl_first_echo, last_pcl_second_echo), axis=1))
                    img.append(last_img)
                    multi_pcl.append(np.asarray(list(p_c2.read_points(msg, field_names=multi_field_names, skip_nans=True)), dtype=np.float32))


                vel_saved = True
                last_pcl_first_echo = np.array([])
                last_pcl_second_echo = np.array([])
                c_saved +=1

            last_multi_pcl = np.asarray(list(p_c2.read_points(msg, field_names=multi_field_names, skip_nans=True)), dtype=np.float32)
            t_last_multi_pcl = t
            t_diff_last_img = t_last_multi_pcl - t_last_img

        assert t_diff_last_img.nsecs <= t_diff_last_pcl.nsecs

    # If last velodyne scan available but not saved yet
    if c_vel_first == c_vel_last and c_vel_first > c_saved:
        pcl.append(np.concatenate((last_pcl_first_echo, last_pcl_second_echo), axis=1))
        img.append(last_img)
        multi_pcl.append(last_multi_pcl)
        c_saved += 1

    output_dataset = osp.join(cfg.RAW_DATA_DIR, dataset)

    # If dataset dir doesnt exists then create it
    if not osp.exists(output_dataset):
        os.makedirs(output_dataset)

    np.save(osp.join(output_dataset, 'images.npy'), img)
    np.save(osp.join(output_dataset, 'pcl.npy'), pcl)
    np.save(osp.join(output_dataset, 'multi_pcl.npy'), multi_pcl)

    print("Dataset saved at " + output_dataset)

    print("c_vel_last %d" % c_vel_last)
    print("c_vel_first %d" % c_vel_first)
    print("c_img %d" % c_img)
    print("c_multi_pcl %d" % c_multi_pcl)
    print("c_saved %d" % c_saved)

    parameters = {
        'count_lidar_last': c_vel_last,
        'count_lidar_first': c_vel_first,
        'count_img': c_img,
        'count_multi_pcl': c_multi_pcl,
        'count_saved': c_saved,
    }

    with open(osp.join(output_dataset, 'config.yaml'), 'w') as f:
        yaml.safe_dump(parameters, f, default_flow_style=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Uncomment this when deploying
    # parser.add_argument('--name', type=str, help='log name')

    # args = parser.parse_args()

    # args.name = 'smoke_car'
    # convert_rosbag(args.name)
    for dataset in datasets:
        convert_rosbag(dataset)
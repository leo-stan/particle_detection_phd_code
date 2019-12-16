import numpy as np
import os.path as osp
import rospy
from cv_bridge import CvBridge
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2 as Pc2
import sensor_msgs.point_cloud2 as p_c2
from sensor_msgs.msg import PointField as Pf
import sys
sys.path.append('../../')
from particle_detection.src.config import cfg
from data_preparation.ros_vis import RosVisualiser
import yaml

# Load data
dataset = '12-smoke_test'
sensor = 'stereo'
display_voxels = True

if sensor == 'lidar_hc':
    dataset_dir = osp.join(cfg.DATASETS_DIR,'hc_datasets', dataset)
elif sensor == 'lidar':
    dataset_dir = osp.join(cfg.DATASETS_DIR, 'dl_datasets', dataset)
else:
    dataset_dir = osp.join(cfg.DATASETS_DIR, 'st_datasets', dataset)

with open(osp.join(dataset_dir, 'config.yaml'), 'r') as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)

rospy.init_node('visualise_labeled_lidar_pcl', anonymous=True)
bridge = CvBridge()

header = Header()
header.frame_id = 'velodyne'

lidar_pub = rospy.Publisher("velodyne_points_labeled", Pc2, queue_size=1)  # Declare publisher
if display_voxels:
    voxel_pub = rospy.Publisher("velodyne_voxels_labeled", Pc2, queue_size=1)
    voxel_header = Header()
    voxel_header.frame_id = 'voxel_map'
    # Declare publisher

p_x = Pf('x', 0, 7, 1)
p_y = Pf('y', 4, 7, 1)
p_z = Pf('z', 8, 7, 1)
p_int = Pf('intensity', 12, 2, 1)
p_echo = Pf('echo', 13, 2, 1)
p_gtlabel = Pf('gt_label', 14, 2, 1)

if sensor == 'lidar_hc' or sensor == 'lidar':
    fields = [p_x, p_y, p_z, p_int,p_echo, p_gtlabel]
else:
    fields = [p_x, p_y, p_z, Pf('rgb', 12, 7, 1), Pf('gt_label', 16, 2, 1)]

if display_voxels:
    voxel_fields = [p_x, p_y, p_z, p_gtlabel]

ros_vis = RosVisualiser(max_id=params['nb_scans'], rate=10, verbose=True)

while ros_vis.state is not 'exit':

    id = ros_vis.update_id()

    pcl = np.load(osp.join(dataset_dir, 'scan_pcls', str(id) + '.npy'))
    header.stamp = rospy.Time.now()
    # pcl = pcl[pcl[:,5] == 1,:]
    lidar_pub.publish(p_c2.create_cloud(header, fields, pcl))
    if display_voxels:
        voxel = np.load(osp.join(dataset_dir, 'scan_voxels', 'coords_' + str(id) + '.npy'))

        voxel = voxel * np.array(
            (params['map_config']['voxel_size_x'], params['map_config']['voxel_size_y'],
             params['map_config']['voxel_size_z']))
        voxel_labels = np.load(osp.join(dataset_dir, 'scan_voxels', 'labels_' + str(id) + '.npy'))

        voxel_header.stamp = header.stamp
        voxel_pub.publish(p_c2.create_cloud(voxel_header, voxel_fields, np.concatenate((voxel, voxel_labels.reshape(-1, 1)), axis=1)))
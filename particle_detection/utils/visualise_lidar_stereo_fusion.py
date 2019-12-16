import numpy as np
import os.path as osp
import os
import rospy
from cv_bridge import CvBridge
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2 as Pc2
import sensor_msgs.point_cloud2 as p_c2
from sensor_msgs.msg import PointField as Pf
import sys
from sensor_msgs.msg import Image

sys.path.append('../..')
from particle_detection.src.config import cfg
from data_preparation.ros_vis import RosVisualiser
import yaml

logdir = cfg.LOG_DIR
# logdir = '/home/leo/hpc-home/phd/particle_detection/particle_detection/logs'

# Load data
sensor = 'lidar' # True if visualising a handcrafted model prediction
custom_threshold = None
display_error = True
display_pcl = False
display_voxel = True
display_img = False

# ensemble = False # True if visualising an ensemble prediction
# model = '20190720_222224_conv_smoke_dust_int_relpos_echo'
# eval_name = 'eval_20190827_194048_visualisation'

# sensor = 'lidar' # True if visualising a handcrafted model prediction
# ensemble = True
# model=''
# eval_name = 'eval_20190906_122133_visualisation'

# sensor = 'stereo' # True if visualising a handcrafted model prediction
# ensemble = True
# model=''
# eval_name = 'eval_20190906_122532_visualisation'

sensor = 'fusion' # True if visualising a handcrafted model prediction
ensemble = False
model=''
eval_name = '20190906_162833_test'

fusion_eval_name = '20190908_155632_visualisation_en5_close'


if display_img:
    dataset = 'visualisation_smoke_dust_int_relpos_echo'
    if sensor == 'lidar':
        data_path = osp.join(cfg.DATASETS_DIR,'dl_datasets',dataset, 'config.yaml')
    elif sensor == 'stereo':
        data_path = osp.join(cfg.DATASETS_DIR,'st_datasets', dataset, 'config.yaml')
    else:
        data_path = osp.join(cfg.DATASETS_DIR, 'hc_datasets', dataset, 'config.yaml')
    with open(data_path, 'r') as f:
        dataset_params = yaml.load(f)

    init_flag = True
    imgs = np.array(())
    for (name, start_id, end_id) in dataset_params['datasets']:
        d_imgs = np.load(osp.join(cfg.RAW_DATA_DIR, name, 'images_sync.npy'))

        if start_id < 0:
            start = 0
        else:
            start = start_id
        if end_id < 0:
             end = d_imgs.shape[0]
        else:
            end = end_id

        imgs_subset = d_imgs[start:end]
        if init_flag:
            imgs = imgs_subset
            init_flag = False
        else:
            imgs = np.concatenate((imgs,imgs_subset))


# ensemble = False # True if visualising an ensemble prediction
# model = '20190615_113731_both_int_relpos_echo'
# eval_name = 'eval_20190702_182745_forest_test_set'

# ensemble = False # True if visualising an ensemble prediction
# model = '20190702_174935_no_forest'
# eval_name = 'eval_20190702_230602_forest_test_set'

# Open set Dust ensemble
# ensemble = True
# eval_name = 'eval_20190702_215304_forest_test_set_en2complementary'


# # Open set Dust single model
# ensemble = False # True if visualising an ensemble prediction
# model = '20190701_174413_smoke_trained'
# eval_name = 'eval_20190704_152222_test_6_dust_cropped'

# # Open set Dust ensemble
# ensemble = True
# eval_name = 'eval_20190703_201325_test_6_dust'

# ensemble = False # True if visualising an ensemble prediction
# model = '20190701_174413_smoke_trained'
# eval_name = 'eval_20190706_131844_smoke_bush_cropped'

if sensor == 'lidar_hc':
    path = os.path.join(logdir, 'hc_logs')
elif sensor == 'stereo':
    path = os.path.join(logdir, 'st_logs')
elif sensor == 'fusion':
    path = os.path.join(logdir, 'fusion_logs')
else:
    path = os.path.join(logdir, 'dl_logs')

if ensemble:
    path = os.path.join(path, 'ensemble_evaluations', eval_name)
elif sensor == 'fusion':
    path = os.path.join(path, eval_name)
else:
    path = os.path.join(path, model, 'evaluations', eval_name)

rospy.init_node('visualise_labeled_lidar_pcl', anonymous=True)
bridge = CvBridge()

header = Header()
header.frame_id = 'velodyne'
voxel_header = Header()
voxel_header.frame_id = 'voxel_map'

lidar_pub = rospy.Publisher("velodyne_points_predicted", Pc2, queue_size=1)  # Declare publisher
voxel_pub = rospy.Publisher("velodyne_voxels_predicted", Pc2, queue_size=1)  # Declare publisher
img_rgb_pub = rospy.Publisher("multisenseS21/left/image_color", Image, queue_size=1)

p_x = Pf('x', 0, 7, 1)
p_y = Pf('y', 4, 7, 1)
p_z = Pf('z', 8, 7, 1)

if sensor == 'stereo':
    p_color = Pf('rgb', 12, 7, 1)
    p_gtlabel = Pf('gt_label', 16, 2, 1)
    p_label = Pf('pred_label', 17, 2, 1)
    fields = [p_x, p_y, p_z, p_color, p_gtlabel, p_label]
    count = 18
else:
    p_int = Pf('intensity', 12, 2, 1)
    p_echo = Pf('echo', 13, 2, 1)
    p_gtlabel = Pf('gt_label', 14, 2, 1)
    p_label = Pf('pred_label', 15, 2, 1)
    fields = [p_x, p_y, p_z, p_int, p_echo, p_gtlabel, p_label]
    count = 16
voxel_fields = [p_x, p_y, p_z, Pf('gt_label', 12, 2, 1), Pf('pred_label', 13, 2, 1)]
voxel_count = 14

if sensor != 'lidar_hc':
    fields.append(Pf('pred_proba_mean', count, 7, 1))
    voxel_fields.append(Pf('pred_proba_mean', voxel_count, 7, 1))
    count += 4
    voxel_count += 4
if ensemble:
    fields.append(Pf('pred_proba_std', count, 7, 1))
    voxel_fields.append(Pf('pred_proba_std', voxel_count, 7, 1))
    count += 4
    voxel_count += 4
if display_error:
    fields.append(Pf('pred_error', count, 2, 1))
    voxel_fields.append(Pf('pred_error', voxel_count, 2, 1))
    count += 1
    voxel_count += 1

rate = rospy.Rate(20)

# for f in filename:
print("loading file...")
if display_pcl:
    pcl = np.load(osp.join(path, 'scans.npy'))
    nb_scans = len(pcl)
else:
    nb_scans = 0
if display_voxel:
    voxel = np.load(osp.join(path, 'scans_voxel.npy'))
    nb_scans = len(voxel)
print("displaying file...")

# for id in range(nb_scans):

ros_vis = RosVisualiser(max_id=nb_scans, rate=20, verbose=True)

while ros_vis.state is not 'exit':

    id = ros_vis.update_id()
    header.stamp = rospy.Time.now()
    voxel_header.stamp = header.stamp
    if display_pcl:
        p = pcl[id]
        if custom_threshold:
            p[:, 6] = (p[:, 7] > custom_threshold).astype(int)
        if display_error:
            if sensor == 'stereo':
                p = np.concatenate((p, (p[:, 4] + p[:, 5] * 2).reshape(-1, 1)), axis=1)
            else:
                p = np.concatenate((p, (p[:, 5] + p[:, 6] * 2).reshape(-1, 1)), axis=1)
        lidar_pub.publish(p_c2.create_cloud(header, fields, p))
    if display_voxel:
        v = voxel[id]
        if custom_threshold:
            v[:, 5] = (v[:, 6] > custom_threshold).astype(int)
        if display_error:
            v = np.concatenate((v, (v[:, 3] + v[:, 4] * 2).reshape(-1, 1)), axis=1)
        voxel_pub.publish(p_c2.create_cloud(voxel_header, voxel_fields, v))
    if display_img:
        img_rgb_pub.publish(bridge.cv2_to_imgmsg(imgs[id], encoding="passthrough"))
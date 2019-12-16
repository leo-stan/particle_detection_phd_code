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
from nav_msgs.msg import OccupancyGrid

sys.path.append('../..')
from particle_detection.src.config import cfg
from data_preparation.ros_vis import RosVisualiser
import yaml

# log_dir = cfg.LOG_DIR
log_dir = '/home/leo/phd/particle_detection/src/particle_detection/particle_detection/logs'
data_dir = '/home/leo/phd/particle_detection/src/particle_detection/particle_detection/data'
raw_data_dir = '/home/leo/phd/particle_detection/src/particle_detection/data'

# Load data
sensor = 'stereo' # True if visualising a handcrafted model prediction
custom_threshold = None
display_error = False
display_pcl = False
display_voxel = True
display_img = True
display_occ_grid = False
occ_grid = 'original'
display_lidar_other = True
display_stereo_other = True

ensemble = False # True if visualising an ensemble prediction
# sensor = 'lidar'
# model = '20190720_222224_conv_smoke_dust_int_relpos_echo'
# eval_name = 'eval_20190827_194048_visualisation'
# eval_name = 'eval_20191006_133450_6_dust_start' # occ grid eval
# eval_name = 'eval_20190722_192652_visualisation' # occ grid eval

# sensor = 'stereo'
# model = '20190809_121706_conv_smoke_dust_relpos_rgb_real1'
# eval_name = 'eval_20190826_221924_visualisation'
#
# model = '20190820_150519_conv_smoke_dust_rgb_real0'
# eval_name = 'eval_20191029_222700_visualisation'

# model = '20190824_115559_conv_smoke_dust_relpos1'
# eval_name = 'eval_20191030_164237_visualisation'

# sensor = 'lidar' # True if visualising a handcrafted model prediction
# ensemble = True
# model=''
# eval_name = 'eval_20190906_122133_visualisation'

# sensor = 'stereo' # True if visualising a handcrafted model prediction
# ensemble = True
# model=''
# eval_name = 'eval_20190906_122532_visualisation'

sensor = 'fusion'
# ensemble = False
model=''
# eval_name = '20190915_123039_visualisation_10en_close'
eval_name = '20191007_153216_visu_40m'

if display_img:
    # dataset = 'visualisation_smoke_dust_int_relpos_echo'
    # dataset = 'visualisation_smoke_dust_relpos_rgb_real'
    dataset = 'visualisation_smoke_dust_int_relpos_echo_fusion_close'
    if sensor == 'lidar':
        data_path = osp.join(data_dir,'dl_datasets',dataset, 'config.yaml')
    elif sensor == 'stereo':
        data_path = osp.join(data_dir,'st_datasets', dataset, 'config.yaml')
    elif sensor == 'fusion': # if fusion provide the lidar dataset for images
        data_path = osp.join(data_dir, 'dl_datasets', dataset, 'config.yaml')
    else:
        data_path = osp.join(data_dir, 'hc_datasets', dataset, 'config.yaml')
    with open(data_path, 'r') as f:
        dataset_params = yaml.load(f)

    init_flag = True
    imgs = np.array(())
    for (name, start_id, end_id) in dataset_params['datasets']:
        d_imgs = np.load(osp.join(raw_data_dir, name, 'images_sync.npy'))

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
    path = os.path.join(log_dir, 'hc_logs')
elif sensor == 'stereo':
    path = os.path.join(log_dir, 'st_logs')
elif sensor == 'fusion':
    path = os.path.join(log_dir, 'fusion_logs/evals')
else:
    path = os.path.join(log_dir, 'dl_logs')

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
voxel_pub = rospy.Publisher("voxels_predicted", Pc2, queue_size=1)  # Declare publisher
lidar_voxel_pub = rospy.Publisher("lidar_voxels_predicted", Pc2, queue_size=1)  # Declare publisher
stereo_voxel_pub = rospy.Publisher("stereo_voxels_predicted", Pc2, queue_size=1)  # Declare publisher
img_rgb_pub = rospy.Publisher("multisenseS21/left/image_color", Image, queue_size=1)
if display_occ_grid:
    og_pub = rospy.Publisher("occupancy_grid", OccupancyGrid, queue_size=1)
    og = OccupancyGrid()
    og.header.frame_id = 'og'
    with open(osp.join(path, 'og_config.yaml'), 'r') as f:
        og_params = yaml.load(f)

    og.info.height = og_params['map_x_size'] / og_params['resolution']
    og.info.width = og_params['map_y_size'] / og_params['resolution']
    og.info.resolution = og_params['resolution']

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

    if sensor == 'fusion':

        voxels = np.load(osp.join(path, 'fused_voxel_proba.npy'))

        lidar_voxels = np.load(osp.join(path, 'lidar_voxel.npy'))
        stereo_voxels = np.load(osp.join(path, 'stereo_voxel.npy'))

        lidar_other = np.load(osp.join(path, 'lidar_voxel_other.npy'))
        for i, lo in enumerate(lidar_other):
            lidar_voxels[i] = np.concatenate((lidar_voxels[i],lo))
            if display_lidar_other:
                voxels[i] = np.concatenate((voxels[i],lo))


        stereo_other = np.load(osp.join(path, 'stereo_voxel_other.npy'))
        for i,so in enumerate(stereo_other):
            stereo_voxels[i] = np.concatenate((stereo_voxels[i], so))
            if display_stereo_other:
                voxels[i] = np.concatenate((voxels[i], so))
    else:
        voxels = np.load(osp.join(path, 'scans_voxel.npy'))
    nb_scans = len(voxels)
if display_occ_grid:
    if occ_grid == 'predicted':
        grids = np.load(osp.join(path, 'pred_maps.npy'))
    elif occ_grid == 'original':
        grids = np.load(osp.join(path, 'original_maps.npy'))
    else: # groundtruth
        grids = np.load(osp.join(path, 'gt_maps.npy'))
print("displaying file...")

# rate = rospy.Rate(10)
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
        v = voxels[id]
        if custom_threshold:
            v[:, 5] = (v[:, 6] > custom_threshold).astype(int)
        if display_error:
            v = np.concatenate((v, (v[:, 3] + v[:, 4] * 2).reshape(-1, 1)), axis=1)
        voxel_pub.publish(p_c2.create_cloud(voxel_header, voxel_fields, v))
        if sensor == 'fusion':
            lidar_voxel_pub.publish(p_c2.create_cloud(voxel_header, voxel_fields, lidar_voxels[id]))
            stereo_voxel_pub.publish(p_c2.create_cloud(voxel_header, voxel_fields, stereo_voxels[id]))
    if display_img:
        img_rgb_pub.publish(bridge.cv2_to_imgmsg(imgs[id], encoding="passthrough"))
    if display_occ_grid:
        og.data = grids[id].T.reshape(-1, 1)[:, 0].tolist()
        # og.data = (np.ones((75, 75), dtype=np.int8)*50).reshape(-1, 1)[:, 0].tolist()
        og.header.stamp = header.stamp
        og_pub.publish(og)
    rate.sleep()
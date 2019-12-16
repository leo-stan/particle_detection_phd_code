"""
Project 3D point clouds in other 3D frame or 2D image
"""

import numpy as np
import transformations as t
import struct

# Load Multisense Husky calibration parameters
t_plate_to_vel = np.asarray([0., 0., 0.3])
theta = np.radians(-90)
r_plate_to_vel = t.rotation_matrix(theta, [0, 0, 1])

T_plate_to_vel = t.concatenate_matrices(t.translation_matrix(t_plate_to_vel), r_plate_to_vel)

T_vel_to_plate = np.matmul(r_plate_to_vel.T, t.translation_matrix(-t_plate_to_vel))
# T_vel_to_plate_inv = np.linalg.inv(T_plate_to_vel)
# r_plate_to_multiS = t.quaternion_matrix([-0.5193485,0.52901035,-0.47704774,0.472070532])

t_plate_to_multiS = np.asarray([0.0909127, 0.152282, -0.0174461])
r_plate_to_multiS = t.concatenate_matrices(t.rotation_matrix(np.radians(-90.873), [0, 0, 1]),
                                           t.rotation_matrix(np.radians(0.244), [0, 1, 0]),
                                           t.rotation_matrix(np.radians(-95.678), [1, 0, 0]))
T_plate_to_multiS = t.concatenate_matrices(t.translation_matrix(t_plate_to_multiS), r_plate_to_multiS)
T_multiS_to_plate = np.matmul(r_plate_to_multiS.T, t.translation_matrix(-t_plate_to_multiS))
# T_multiS_to_plate_inv = np.linalg.inv(T_plate_to_multiS)

T_multiS_to_vel = t.concatenate_matrices(T_multiS_to_plate, T_plate_to_vel)
T_vel_to_multiS = t.concatenate_matrices(T_vel_to_plate, T_plate_to_multiS)
# T_vel_to_multiS_inv = np.linalg.inv(T_multiS_to_vel)

CM = np.matmul(np.array([455.1, 0, 521.408, 0, 455.48175, 271.607177, 0, 0, 1.0]).reshape(3, 3),
               np.concatenate((np.eye(3), np.zeros(3).reshape(3, 1)), axis=1))


# DC = [-0.043304, 0.097604, 0.000498, 0.00322, -0.0476361]


def project_scan(img, lidar_pcl=None, multi_pcl=None, project_img=False, save_coord=False):
    # LIDAR
    if isinstance(lidar_pcl, np.ndarray):
        cropped_pcl = np.array([]).reshape(0, lidar_pcl.shape[1])
        for p in lidar_pcl:
            if p[1] > 0:
                pp = np.matmul(np.matmul(CM, T_multiS_to_vel), np.append(p[:3], 1))
                pp = np.floor(pp / pp[2]).astype(int)[:2]
                if 0 < pp[1] < img.shape[0] and 0 < pp[0] < img.shape[1]:
                    cropped_pcl = np.concatenate((cropped_pcl, p.reshape(1, lidar_pcl.shape[1])))
    else:
        cropped_pcl = None

    # MULTISENSE
    if isinstance(multi_pcl, np.ndarray):

        img_labels = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
        img_depth = np.zeros((img.shape[0], img.shape[1], 1))
        img_rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        multi_proj_in_lidar_frame = multi_pcl.copy()
        multi_pcl_coord = np.zeros((multi_pcl.shape[0], 2), dtype=np.uint16)
        # Multisense in img
        if project_img or save_coord:
            for i, p in enumerate(multi_pcl):
                pp_3d = np.matmul(T_multiS_to_vel, np.append(p[:3], 1))
                pp_2d = np.matmul(CM, pp_3d)
                pp_2d = np.round(pp_2d / pp_2d[2]).astype(int)[:2]
                if 0 < pp_2d[1] < img.shape[0] and 0 < pp_2d[0] < img.shape[1]:


                    # Manual offset to remove error in reprojection
                    offset_x = 0
                    if 7 < pp_2d[1]:
                        offset_x += 1
                    if 133 < pp_2d[1]:
                        offset_x += 1
                    if 258 < pp_2d[1]:
                        offset_x += 1
                    if 384 < pp_2d[1]:
                        offset_x += 1
                    if 509 < pp_2d[1]:
                        offset_x += 1

                    offset_y = 0
                    if 114 < pp_2d[0]:
                        offset_y += 1
                    if 254 < pp_2d[0]:
                        offset_y += 1
                    if 394 < pp_2d[0]:
                        offset_y += 1
                    if 534 < pp_2d[0]:
                        offset_y += 1
                    if 674 < pp_2d[0]:
                        offset_y += 1
                    if 815 < pp_2d[0]:
                        offset_y += 1
                    if 955 < pp_2d[0]:
                        offset_y += 1
                    img_labels[pp_2d[1] - offset_x, pp_2d[0] - offset_y, 0] = p[4]*85+85
                    img_depth[pp_2d[1] - offset_x, pp_2d[0] - offset_y, 0] = pp_3d[2]
                    buffer = struct.pack('f', p[3])
                    r, g, b, _ = struct.unpack('bbbb', buffer)
                    img_rgb[pp_2d[1] - offset_x, pp_2d[0] - offset_y, :] = np.array([r, g, b], dtype=np.uint8)
                    if save_coord:
                        multi_pcl_coord[i, 0] = pp_2d[1]
                        multi_pcl_coord[i, 1] = pp_2d[0]

        else:
            for i, p in enumerate(multi_pcl):
                # Multisense in lidar frame
                pp = np.matmul(T_vel_to_multiS, np.append(p[:3], 1))
                multi_proj_in_lidar_frame[i, :3] = pp[:3]
    else:
        img_labels = None
        img_depth = None
        img_rgb = None
        multi_proj_in_lidar_frame = None
        multi_pcl_coord = None
    return [cropped_pcl, img_labels, img_depth, img_rgb, multi_proj_in_lidar_frame, multi_pcl_coord]

import numpy as np
from sklearn.decomposition import PCA
import math
import struct


def compute_hc_features(features, voxel, mask):
    voxel_features = np.array([])
    voxel = voxel[mask.reshape(mask.shape[0]), :]

    if 'roughness' in features:
        # Need at least 3 points to compute PCA
        if voxel.shape[0] > 3:
            pca = PCA()
            pca.fit(voxel[:, :3])
            voxel_features = np.concatenate((voxel_features, np.array([pca.singular_values_[2]])))
        else:
            voxel_features = np.concatenate((voxel_features, np.array([0])))
    if 'slope' in features:
        # Need at least 3 points to compute PCA
        if voxel.shape[0] > 3:
            if 'roughness' not in features:
                pca = PCA()
                pca.fit(voxel[:, :3])
            # 0 is any vector on the xy plane, pi/2 is a vector along the z axis
            voxel_features = np.concatenate((voxel_features, np.array([abs(math.asin(pca.components_[2, 2]))])))
        else:
            voxel_features = np.concatenate((voxel_features, np.array([0])))
    if 'intmean' in features:
        voxel_features = np.concatenate((voxel_features, np.array([np.mean(voxel[:, 3])])))
    if 'intvar' in features:
        voxel_features = np.concatenate((voxel_features, np.array([np.std(voxel[:, 3])])))

    if 'echo' in features:
        unique, cnts = np.unique(voxel[:, 4], return_counts=True)
        echo = unique[np.argmax(cnts)]
        # One hot encoding: concatenate three times to have three individual vectors for 0 1 2
        if echo == 0:
            voxel_features = np.concatenate((voxel_features, np.array([1, 0, 0])))
        elif echo == 1:
            voxel_features = np.concatenate((voxel_features, np.array([0, 1, 0])))
        else:
            voxel_features = np.concatenate((voxel_features, np.array([0, 0, 1])))

    return voxel_features


def extract_scan(raw_lidar_pcl, features, map_config, sensor):
    """
    Puts raw lidar points into voxels and prepare data for training
    :param raw_lidar_pcl: numpy array with raw lidar points
    :return:
    """

    # if shuffle:
    #     np.random.shuffle(raw_lidar_pcl)

    # Process raw scan
    # Apply -90deg rotation on z axis to go from robot to map
    # raw_lidar_pcl[:, [0, 1]] = raw_lidar_pcl[:, [1, 0]]  # Swap x, y axis
    # raw_lidar_pcl[:, :3] = raw_lidar_pcl[:, :3] * np.array([-1, 1, 1], dtype=np.int8)

    # removes points at [0,0,0]
    # if np.any(np.sum(raw_lidar_pcl[:, :3], axis=1) == 0):
    raw_lidar_pcl = raw_lidar_pcl[np.sum(raw_lidar_pcl[:, :3], axis=1) != 0, :]
    # Translation from map to robot_footprint
    husky_footprint_coord = np.array(
        [map_config['map_to_vel_x'], map_config['map_to_vel_y'], map_config['map_to_vel_z']], dtype=np.float32)

    # Lidar points in map coordinate
    shifted_coord = raw_lidar_pcl[:, :3] + husky_footprint_coord

    voxel_size = np.array([map_config['voxel_size_x'], map_config['voxel_size_y'], map_config['voxel_size_z']],
                          dtype=np.float32)
    grid_size = np.array(
        [map_config['map_size_x'] / map_config['voxel_size_x'], map_config['map_size_y'] / map_config['voxel_size_y'],
         map_config['map_size_z'] / map_config['voxel_size_z']], dtype=np.int64)

    voxel_index = np.floor(shifted_coord / voxel_size).astype(np.int)

    bound_x = np.logical_and(
        voxel_index[:, 0] >= 0, voxel_index[:, 0] < grid_size[0])
    bound_y = np.logical_and(
        voxel_index[:, 1] >= 0, voxel_index[:, 1] < grid_size[1])
    bound_z = np.logical_and(
        voxel_index[:, 2] >= 0, voxel_index[:, 2] < grid_size[2])

    bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    # Raw scan within bounds
    raw_lidar_pcl = raw_lidar_pcl[bound_box]
    pcl_features = raw_lidar_pcl.copy()
    voxel_index = voxel_index[bound_box]
    if "vox_pos" in features:
        pcl_features[:, :3] = (pcl_features[:, :3] + husky_footprint_coord) - voxel_index * voxel_size

    # [K, 3] coordinate buffer as described in the paper
    coordinate_buffer = np.unique(voxel_index, axis=0)

    # Number of voxels in scan
    K = len(coordinate_buffer)
    # Max number of lidar points in each voxel
    T = map_config['voxel_pt_count']

    # [K, 1] store number of points in each voxel grid
    number_buffer = np.zeros(shape=K, dtype=np.int64)

    # [K, T, 8] feature buffer as described in the paper
    if sensor == 'stereo':
        feature_buffer = np.zeros(shape=(K, T, 5), dtype=np.float32)
    else:
        feature_buffer = np.zeros(shape=(K, T, 6), dtype=np.float32)
    label_buffer = np.zeros(shape=K, dtype=np.uint8)
    # build a reverse index for coordinate buffer
    index_buffer = {}
    for i in range(K):
        index_buffer[tuple(coordinate_buffer[i])] = i

    for voxel, point in zip(voxel_index, pcl_features):
        index = index_buffer[tuple(voxel)]
        number = number_buffer[index]
        if number < T:
            feature_buffer[index, number, :] = point
            number_buffer[index] += 1
            labels, counts = np.unique(feature_buffer[index, :number_buffer[index], -1], return_counts=True)
            # Label is the maximum number of particle vs non particle
            label_buffer[index] = labels[np.argmax(counts)].astype(int)

    # Create a mask to only populate number of points in the voxel and not all T points
    mask = ~np.all(feature_buffer[:, :, :] == 0, axis=2)
    mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))

    if sensor == 'lidar_hc':
        init_flag = False
        # Compute features
        for i in range(K):
            if not init_flag:
                # Pick and choose features here
                selected_features = compute_hc_features(features, feature_buffer[i, :, :], mask[i, :, :])
                selected_buffer = np.zeros((feature_buffer.shape[0],selected_features.shape[0]), dtype=np.float32)
                selected_buffer[i, :] = selected_features
                init_flag = True
            else:
                selected_buffer[i, :] = compute_hc_features(features, feature_buffer[i, :, :], mask[i, :, :])

    elif sensor == 'lidar':
        # Pick and choose features here
        selected_buffer = np.array([], dtype=np.float32).reshape(feature_buffer.shape[0], feature_buffer.shape[1], 0)

        if "pos" in features or "vox_pos" in features:
            selected_buffer = np.concatenate((selected_buffer, feature_buffer[:, :, :3]), axis=2)
        if "int" in features:
            selected_buffer = np.concatenate((selected_buffer, feature_buffer[:, :, 3:4]), axis=2)
            # Add Gaussian noise to intensity
            # selected_buffer[:, :, -1:] = (selected_buffer[:, :, -1:]+np.random.normal(0,3,(selected_buffer.shape[0],selected_buffer.shape[1],1)))*mask
        # Old echo not one-hot encoded
        # if "echo" in features:
        #     selected_buffer = np.concatenate((selected_buffer, feature_buffer[:, :, 4:5]), axis=2)
        if "echo" in features:
            # If one hot echo, concatenate three times to have three individual vectors for 0 1 2
            selected_buffer = np.concatenate((selected_buffer, feature_buffer[:, :, 4:5]), axis=2) # This needs to be 4:5 and not 4 to keep shape
            selected_buffer[:, :, -1:] = (selected_buffer[:, :, -1:] == 0) * mask
            selected_buffer = np.concatenate((selected_buffer, feature_buffer[:, :, 4:5]), axis=2)
            selected_buffer[:, :, -1:] = (selected_buffer[:, :, -1:] == 1) * mask
            selected_buffer = np.concatenate((selected_buffer, feature_buffer[:, :, 4:5]), axis=2)
            selected_buffer[:, :, -1:] = (selected_buffer[:, :, -1:] == 2) * mask

        if "relpos" in features:
            # Compute relative position
            relpos_buffer = np.zeros((K, T, 3))
            for i in range(K):
                relpos_buffer[i, :number_buffer[i], :] = feature_buffer[i, :number_buffer[i], :3] - feature_buffer[i, :number_buffer[i], :3].sum(axis=0, keepdims=True) / number_buffer[i]
            selected_buffer = np.concatenate((selected_buffer, relpos_buffer), axis=2)

    else:
        selected_buffer = np.array([], dtype=np.float32).reshape(feature_buffer.shape[0], feature_buffer.shape[1], 0)

        if "relpos" in features:
            # Compute relative position
            relpos_buffer = np.zeros((K, T, 3))
            for i in range(K):
                relpos_buffer[i, :number_buffer[i], :] = feature_buffer[i, :number_buffer[i], :3] - feature_buffer[i, :number_buffer[i],:3].sum(axis=0,keepdims=True) / number_buffer[i]
            selected_buffer = np.concatenate((selected_buffer, relpos_buffer), axis=2)

        if "rgb" in features:
            color_buffer = np.zeros((K, T, 3)).astype(np.uint8)
            for i in range(K):
                for j in range(number_buffer[i]):
                    buffer = struct.pack('f', feature_buffer[i, j, 3])
                    r, g, b, _ = struct.unpack('bbbb', buffer)
                    color_buffer[i, j, :] = np.array([r, g, b], dtype=np.uint8)

            selected_buffer = np.concatenate((selected_buffer, color_buffer), axis=2)

    return selected_buffer, coordinate_buffer, label_buffer, raw_lidar_pcl

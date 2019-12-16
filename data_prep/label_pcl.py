#!/usr/bin/env python

import argparse
import multiprocessing
from joblib import Parallel, delayed
import os.path as osp
import numpy as np
import tqdm
import sys
sys.path.append('../')
from particle_detection.src.config import cfg
import yaml

datasets_dir = cfg.RAW_DATA_DIR


def update_ref_map(ref_map, pcl, map_config):
    vox_map, _, _, _ = compute_voxel_map(pcl, map_config)
    ref_map = np.maximum(ref_map, vox_map)
    return ref_map

def label_voxel_map(ref_map, pcl, map_config, particle_label, crf=True):
    vox_map, pcl_in, voxel_index, pcl_out = compute_voxel_map(pcl, map_config)

    diff_map = ref_map+(vox_map*2)
    grid_size = np.array(
        [map_config['map_size_x'] / map_config['voxel_size_x'], map_config['map_size_y'] / map_config['voxel_size_y'],
         map_config['map_size_z'] / map_config['voxel_size_z']], dtype=np.int64)
    vox_map = np.zeros(grid_size, dtype=np.uint8)

    # Conditional Random Field
    if crf:
        for i in range(diff_map.shape[0]):
            for j in range(diff_map.shape[1]):
                for k in range(diff_map.shape[2]):
                    if diff_map[i,j,k] == 2 or diff_map[i,j,k] == 3:
                        # Check all neighbours and apply majority
                        nb_non_particle_neighbours = 0
                        nb_particle_neighbours = 0
                        for l in range(-1,1):
                            for m in range(-1, 1):
                                for n in range(-1, 1):
                                    # Check cell is in map
                                    if 0 < i+l < grid_size[0] and 0 < j+m < grid_size[1] and 0 < k+n < grid_size[2]:
                                        if diff_map[i,j,k] == 2:
                                            nb_particle_neighbours+=1
                                        elif diff_map[i,j,k] == 3:
                                            nb_non_particle_neighbours += 1
                        # If cell is particle and most neighbours are non-particle then change label
                        if diff_map[i,j,k] == 2 and nb_non_particle_neighbours > nb_particle_neighbours:
                            diff_map[i, j, k] = 3
                        # # If cell is non-particle and most neighbours are particle then change label
                        # if diff_map[i, j, k] == 3 and nb_non_particle_neighbours < nb_particle_neighbours:
                        #     diff_map[i, j, k] = 2

    # points outside of voxel map
    labelled_pcl_out = np.concatenate((pcl_out, np.zeros((pcl_out.shape[0], 1))), axis=1)

    # points inside voxel map
    labelled_pcl_in = np.concatenate((pcl_in, np.zeros((pcl_in.shape[0], 1))), axis=1)
    # Grab the voxels that are present in vox_map but not in ref map (particles)
    labelled_pcl_in[diff_map[voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]] == 2, 4] = particle_label

    return np.concatenate((labelled_pcl_in, labelled_pcl_out))

def compute_voxel_map(pcl, map_config):

    # Translation from map to robot_footprint
    husky_footprint_coord = np.array(
        [map_config['map_to_vel_x'], map_config['map_to_vel_y'], map_config['map_to_vel_z']], dtype=np.float32)

    # Lidar points in map coordinate
    shifted_coord = pcl[:, :3] + husky_footprint_coord

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
    pcl_in = pcl[bound_box]
    pcl_out = pcl[np.logical_not(bound_box)]
    voxel_index = voxel_index[bound_box]

    vox_map = np.zeros(grid_size, dtype=np.uint8)
    vox_map[voxel_index[:, 0],voxel_index[:, 1], voxel_index[:, 2]] = 1

    return vox_map, pcl_in, voxel_index, pcl_out


def label_pcls(name, ref_id, sampled=False):
    # 1 load pcl
    if sampled:
        pcls = np.load(osp.join(datasets_dir, name, 'multi_pcl_sync_sampled.npy'))
    else:
        pcls = np.load(osp.join(datasets_dir, name, 'multi_pcl_sync.npy'))
    labelled_pcls = pcls.copy()

    # Load dataset parameters
    with open(osp.join(datasets_dir, name, 'config.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    config['map_config'] = {
            'map_size_x': 20,
            'map_size_y': 5,
            'map_size_z': 10,
            'voxel_size_x': 0.3,
            'voxel_size_y': 0.3,
            'voxel_size_z': 0.3,
            'map_to_vel_x': 10,
            'map_to_vel_y': 4,
            'map_to_vel_z': 0
        }
    config['ref_split'] = ref_id
    config['crf'] = True

    with open(osp.join(datasets_dir, name,  'config.yaml'), 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    # 2 compute reference map (extract_scan)
    ref_map = np.zeros((
            int(config['map_config']['map_size_x']/config['map_config']['voxel_size_x']),
            int(config['map_config']['map_size_y']/config['map_config']['voxel_size_y']),
            int(config['map_config']['map_size_z']/config['map_config']['voxel_size_z'])), dtype=np.uint8)

    if 'smoke' in name:
        particle_label = 2
    else:
        particle_label = 1

    for i in range(config['ref_split']):
        pcl = pcls[i]
        ref_map = update_ref_map(ref_map, pcl, config['map_config'])
        labelled_pcls[i] = np.concatenate((labelled_pcls[i], np.zeros((labelled_pcls[i].shape[0], 1))), axis=1)

    # 3 parallelised comparison of each subsequent scans
    for i in tqdm.tqdm(range(config['ref_split'], len(pcls))):
        labelled_pcls[i] = label_voxel_map(ref_map, pcls[i], config['map_config'], particle_label, crf=config['crf'])

    if sampled:
        print('labelled sampled pcl saved')
        np.save(osp.join(datasets_dir, name, 'multi_pcl_sync_sampled_labelled.npy'), labelled_pcls)
    else:
        print('labelled pcl saved')
        np.save(osp.join(datasets_dir, name, 'multi_pcl_sync_labelled.npy'), labelled_pcls)

    # num_cores = multiprocessing.cpu_count()
    # results = Parallel(n_jobs=num_cores)(
    #     delayed(extract_scan)(pcl, parameters['features'], parameters['map_config'], hand_crafted) for pcl in tqdm.tqdm(
    #         pcls, total=pcls.shape[0], desc='Generating Voxel Data:', ncols=80, leave=False))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Uncomment this when deploying
    parser.add_argument('--name', type=str, help='datset name')
    parser.add_argument('--sampled', type=str, default='True', help='sampled flag')

    args = parser.parse_args()
    print('Label pcl dataset: %s' % args.name)
    sampled = args.sampled == 'True'
    if sampled:
        print('sampled')
    # args.name = '12-smoke'
    args.split_id = 5
    # args.sampled = True
    label_pcls(args.name, args.split_id, sampled=sampled)

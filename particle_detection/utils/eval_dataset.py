#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename: eval_dataset.py
Author: Leo Stanislas
Date Created: 26 Aug. 2019
Description: Provide a summary of the dataset
"""

# No of particles points
# No of non-particle points
# Fog/Dust point ratio
# Particle Voxels
# Non-particle Voxels
# No of images/scans


import sys

sys.path.insert(0, '../../')


import os.path as osp
from particle_detection.src.config import cfg
import yaml
import numpy as np
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--sensor', type=str, help='sensor')
    parser.add_argument('--dataset', type=str, help='name of dataset to eval')

    args = parser.parse_args()
    # Load data
    sensor = args.sensor
    dataset_name = args.dataset

    if sensor == 'lidar':
        datasets_dir = osp.join(cfg.DATASETS_DIR, 'dl_datasets')
        logs_dir = osp.join(cfg.LOG_DIR, 'dl_logs')
    else:
        datasets_dir = osp.join(cfg.DATASETS_DIR, 'st_datasets')
        logs_dir = osp.join(cfg.LOG_DIR, 'st_logs')

    # Load dataset parameters
    with open(osp.join(datasets_dir, dataset_name, 'config.yaml'), 'r') as f:
        data_params = yaml.load(f, Loader=yaml.SafeLoader)

    nb_particle_pts = 0
    nb_non_particle_pts = 0

    nb_particle_vox = 0
    nb_non_particle_vox = 0

    for i in range(data_params['nb_scans']):
        voxel_labels = np.load(osp.join(datasets_dir, dataset_name,'scan_voxels','labels_'+str(i)+'.npy'))
        pcl = np.load(osp.join(datasets_dir, dataset_name, 'scan_pcls', str(i)+'.npy'))
        nb_particle_vox += np.sum((voxel_labels==1).astype(int))
        nb_non_particle_vox += np.sum((voxel_labels==0).astype(int))
        if sensor == 'lidar_hc' or sensor == 'lidar':
            nb_particle_pts += np.sum((pcl[:, 5] == 1).astype(int))
            nb_non_particle_pts += np.sum((pcl[:, 5] == 0).astype(int))
        else:
            nb_particle_pts += np.sum((pcl[:, 4] == 1).astype(int))
            nb_non_particle_pts += np.sum((pcl[:, 4] == 0).astype(int))

    with open(osp.join(datasets_dir, dataset_name, 'summary.txt'), 'w') as f:
        f.write('nb_particle_pts: %s\n' % nb_particle_pts)
        f.write('nb_non_particle_pts: %s\n' % nb_non_particle_pts)
        f.write('nb_particle_vox: %s\n' % nb_particle_vox)
        f.write('nb_non_particle_vox: %s\n' % nb_non_particle_vox)
        f.write('nb_scans: %s\n' % data_params['nb_scans'])
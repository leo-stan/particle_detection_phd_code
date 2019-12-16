
"""
Crop LiDAR point cloud and transform multisense point cloud in LiDAR frame to refine both labels
"""

import numpy as np
import sys
import os.path as osp

from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
sys.path.insert(0, '..')
from particle_detection.src.config import cfg
import argparse
from project_scan import project_scan

datasets_dir = cfg.RAW_DATA_DIR


def project_pcls(dataset, multisense_proj, multisense_sampled, lidar_crop, parallel_proc=True):
    print('Processing dataset: %s' % dataset)

    imgs = np.load(osp.join(datasets_dir, dataset, 'images_sync.npy'))
    if lidar_crop:
        lidar_pcls = np.load(osp.join(datasets_dir, dataset, 'pcl_labeled_spaces_converted_sync.npy'))

    if multisense_proj:
        if multisense_sampled:
            multi_pcls = np.load(osp.join(datasets_dir, dataset, 'multi_pcl_sync_sampled_labelled.npy'))
        else:
            multi_pcls = np.load(osp.join(datasets_dir, dataset, 'multi_pcl_sync_labelled.npy'))

    print("Data loaded")

    # Perform Projection
    if multisense_proj and lidar_crop:
        if parallel_proc:
            num_cores = multiprocessing.cpu_count()
            results = Parallel(n_jobs=num_cores)(delayed(project_scan)(i, p, m) for i, p, m in
                                                 tqdm(zip(imgs, lidar_pcls, multi_pcls)))
        else:
            results = []
            for i, p, m in tqdm(zip(imgs, lidar_pcls, multi_pcls)):
                results = results.append(project_scan(i, p, m))
    elif multisense_proj and not lidar_crop:
        if parallel_proc:
            num_cores = multiprocessing.cpu_count()
            results = Parallel(n_jobs=num_cores)(delayed(project_scan)(i, None, m) for i, m in
                                                 tqdm(zip(imgs, multi_pcls)))
        else:
            results = []
            for i, m in tqdm(zip(imgs, multi_pcls)):
                results.append(project_scan(i, None, m))
    elif lidar_crop and not multisense_proj:
        if parallel_proc:
            num_cores = multiprocessing.cpu_count()
            results = Parallel(n_jobs=num_cores)(delayed(project_scan)(i, p, None) for i, p in
                                                 tqdm(zip(imgs, lidar_pcls)))
        else:
            results = []
        for i, p in tqdm(zip(imgs, lidar_pcls)):
            results = results.append(project_scan(i, p, None))
    else:
        print('Need to select at least multisense or lidar...')
    # #
    results = np.asarray(results)

    if lidar_crop:
        np.save(osp.join(datasets_dir, dataset, 'pcl_labeled_spaces_converted_sync_cropped.npy'), results[:, 0])
    if multisense_proj:
        if multisense_sampled:
            np.save(osp.join(datasets_dir, dataset, 'multi_pcl_sync_sampled_labelled_projected.npy'), results[:, 4])
        else:
            np.save(osp.join(datasets_dir, dataset, 'multi_pcl_sync_labelled_projected.npy'), results[:, 4])

    print("projected point clouds saved")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Uncomment this when deploying
    parser.add_argument('--name', type=str, help='dataset name')
    parser.add_argument('--sampled', type=str, default='True', help='sampled flag')
    parser.add_argument('--lidar_crop', type=str, default='True', help='crop lidar flag')

    args = parser.parse_args()

    # Arguments
    # datasets = [
    #     # '1-dust',
    #     # '2-dust',
    #     # '3-dust',
    #     # '4-dust',
    #     # '5-dust',
    #     # '6-dust',
    #     # '7-dust',
    #     # '8-dust',
    #     # '9-smoke',
    #     # '10-smoke',
    #     # '11-smoke',
    #     '12-smoke',
    #     # '13-smoke',
    #     # '14-smoke',
    #     # '15-smoke',
    #     # '16-smoke',
    #     # '17-smoke',
    #     # '18-smoke',
    #     # '19-smoke',
    #     # 'smoke_bush'
    # ]
    print('project pcl %s' % args.name)
    if args.sampled:
        print('sampled')
    multisense_proj = True
    sampled = args.sampled == 'True'
    lidar_crop = args.lidar_crop == 'True'
    # args.sampled = True
    # args.lidar_crop = True

    # for dataset in datasets:
    project_pcls(args.name, multisense_proj, sampled, lidar_crop, parallel_proc=True)

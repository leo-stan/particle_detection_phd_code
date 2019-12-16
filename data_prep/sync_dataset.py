#!/usr/bin/env python

# This scripts remove the velodyne scans than don't have a corresponding multisense image. They are originally kept for labeling (the first few scans are important for background substraction)
import sys
sys.path.insert(0, '../')

import numpy as np
import os.path as osp
from particle_detection.src.config import cfg
import yaml
import argparse

datasets_dir = cfg.RAW_DATA_DIR

def sync_dataset(dataset):
    print('Synchronising dataset %s' % dataset)

    # Load dataset parameters
    with open(osp.join(datasets_dir, dataset, 'config.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    lidar_pcls = np.load(osp.join(datasets_dir, dataset, 'pcl_labeled_spaces_converted.npy'))
    imgs = np.load(osp.join(datasets_dir, dataset, 'images.npy'))
    multi_pcls = np.load(osp.join(datasets_dir, dataset, 'multi_pcl.npy'))
    c = 0

    print('Initial size: %d' % config['count_saved'])

    # Count number of empty scans at the start (padding to keep all lidar scans for labeling)
    for i in range(config['count_saved']):
        if ~imgs[i].any() or ~multi_pcls[i].any() or ~lidar_pcls[i].any():
            c += 1
    imgs = imgs[c:config['count_saved'], :, :, :]
    multi_pcls = multi_pcls[c:config['count_saved']]
    lidar_pcls = lidar_pcls[c:config['count_saved']]

    print('Final size: %d' % (config['count_saved']-c))

    config['count_sync'] = config['count_saved']-c
    config['img_size_x'] = imgs[0].shape[0]
    config['img_size_y'] = imgs[0].shape[1]

    with open(osp.join(datasets_dir, dataset, 'config.yaml'), 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    np.save(osp.join(datasets_dir, dataset, 'multi_pcl_sync.npy'), multi_pcls)
    np.save(osp.join(datasets_dir, dataset, 'pcl_labeled_spaces_converted_sync.npy'), lidar_pcls)
    np.save(osp.join(datasets_dir, dataset, 'images_sync.npy'), imgs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Uncomment this when deploying
    parser.add_argument('--name', type=str, help='dataset name')

    args = parser.parse_args()

    # args.name = '10-smoke'
    print('sync dataset %s' % args.name)

    # for dataset in datasets:
    sync_dataset(args.name)
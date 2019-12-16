import sys
sys.path.append('../')

import numpy as np
import os.path as osp
from particle_detection.src.config import cfg
import tqdm
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Uncomment this when deploying
    parser.add_argument('--name', type=str, help='dataset name')
    # parser.add_argument('--split_id', type=int, help='split id')

    args = parser.parse_args()

    # new_ratio = 0.075
    new_ratio = 0.15
    multi_pcl = np.load(osp.join(cfg.RAW_DATA_DIR, args.name, 'multi_pcl_sync.npy'))

    new_pcl = []

    for scan in tqdm.tqdm(multi_pcl, desc='Sampling multisense pointcloud'):

        new_length = int(scan.shape[0]*new_ratio)
        np.random.shuffle(scan)
        new_pcl.append(scan[:new_length, :])

    np.save(osp.join(cfg.RAW_DATA_DIR, args.name,'multi_pcl_sync_sampled.npy'), np.asarray(new_pcl))

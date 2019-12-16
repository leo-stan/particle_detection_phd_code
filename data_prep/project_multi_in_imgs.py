
"""
Projects multisense points into images after refining labels
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


def project_multi_in_imgs(dataset, parallel_proc=True, sampled=False, save_coord=False):
    print('Processing dataset: %s' % dataset)

    imgs = np.load(osp.join(datasets_dir, dataset, 'images_sync.npy'))
    if sampled:
        multi_pcls = np.load(osp.join(datasets_dir, dataset, 'multi_pcl_sync_sampled_labelled_projected_refined.npy'))
    else:
        multi_pcls = np.load(
            osp.join(datasets_dir, dataset, 'multi_pcl_sync_labelled_projected_refined.npy'))
    print("Data loaded")

    # Perform Projection

    if parallel_proc:
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(project_scan)(i, None, m, True, save_coord) for i, m in
                                             tqdm(zip(imgs, multi_pcls)))
    else:
        results = []
        for i, m in tqdm(zip(imgs, multi_pcls)):
            results.append(project_scan(i, None, m, True, save_coord))

    #
    results = np.asarray(results)
    if not sampled:
        np.save(osp.join(datasets_dir, dataset, 'images_sync_label.npy'), results[:, 1])
        np.save(osp.join(datasets_dir, dataset, 'images_sync_depth.npy'), results[:, 2])
        np.save(osp.join(datasets_dir, dataset, 'images_sync_rgb.npy'), results[:, 3])
    if save_coord:
        if sampled:
            np.save(osp.join(datasets_dir, dataset, 'multi_pcl_sync_sampled_img_coord.npy'), results[:, 5])
        else:
            np.save(osp.join(datasets_dir, dataset, 'multi_pcl_sync_img_coord.npy'), results[:, 5])

    print("projected imgs saved")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Uncomment this when deploying
    parser.add_argument('--name', type=str, help='dataset name')

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

    # for dataset in datasets:
    # args.name = '12-smoke'
    project_multi_in_imgs(args.name, sampled=True, save_coord=True)

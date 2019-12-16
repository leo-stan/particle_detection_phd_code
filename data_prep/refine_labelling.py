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

datasets_dir = cfg.RAW_DATA_DIR


def refine_pcl(pcl, scan_id, name, sensor):
    refined_pcl = pcl.copy()

    if 'smoke' in name:
        particle_label = 2
    else:
        particle_label = 1

    for p in refined_pcl:
        x = p[0]
        y = p[1]
        z = p[2]
        force_particle = False
        force_non_particle = False

        if '1-dust' == name:
            if sensor == 'stereo':
                y1 = 3.27
                x1 = 1.2
                y2 = 6.76
                x2 = 4.6
                force_non_particle = (y < y1 + (y2 - y1) / (x2 - x1) * (x - x1)) or z < -1.12 or (x*x +y*y) > 64 or x < -3.4
            else:
                # Ground removal
                y1 = 2.29
                z1 = -1.36
                y2 = 8.48
                z2 = -1.23
                force_non_particle = z <= z1 + (z2 - z1) / (y2 - y1) * (y - y1)

        if '3-dust' == name:
            if sensor == 'stereo':
                y1 = 3.27
                x1 = 1.2
                y2 = 6.76
                x2 = 4.6
                force_non_particle = (y < y1 + (y2 - y1) / (x2 - x1) * (x - x1)) or z < -1.12 or (x*x +y*y) > 64 or x < -3.4 or x > 2
                z3 = -1.26
                y3 = 1.51
                z4 = -1.00
                y4 = 10.76
                force_particle = z > z3 + (z4 - z3) / (y4 - y3) * (y - y3) and -1.5 < x < 1 and y < 8

            else:
                # Ground removal
                y1 = 2.29
                z1 = -1.36
                y2 = 8.48
                z2 = -1.23
                force_non_particle = z <= z1 + (z2 - z1) / (y2 - y1) * (y - y1)

        if '2-dust' == name :
            if sensor == 'stereo':
                y1 = 3.27
                x1 = 1.2
                y2 = 6.76
                x2 = 4.6
                force_non_particle = (y < y1 + (y2 - y1) / (x2 - x1) * (x - x1)) or z < -1.12 or (x*x +y*y) > 64 or -3.4 > x or x > 0.5
            else:
                # Ground removal
                y1 = 2.29
                z1 = -1.36
                y2 = 8.48
                z2 = -1.23
                force_non_particle = z <= z1 + (z2 - z1) / (y2 - y1) * (y - y1)
        if '4-dust' == name :
            if sensor == 'stereo':
                # Ground removal
                y1 = 1.55
                z1 = -1.30
                y2 = 13.28
                z2 = -1.10
                force_non_particle = z <= z1 + (z2 - z1) / (y2 - y1) * (y - y1) or x < -0.5 or x > 3 or y > 7.5

        if ('5-dust' == name or '6-dust' == name or '7-dust' == name) and sensor == 'stereo':
            force_non_particle = True

            # if sensor == 'stereo':
            #     y1 = 3.27
            #     x1 = 1.2
            #     y2 = 6.76
            #     x2 = 4.6
            #     force_non_particle = (y < y1 + (y2 - y1) / (x2 - x1) * (x - x1)) or z < -1.12 or (
            #                 x * x + y * y) > 64 or x < -3.4
            # else:
            #     # Ground removal
            #     y1 = 2.29
            #     z1 = -1.36
            #     y2 = 8.48
            #     z2 = -1.23
            #     force_non_particle = z <= z1 + (z2 - z1) / (y2 - y1) * (y - y1)

        if '12-smoke' == name:
            force_particle = (x < 2 and y < 3 and z > -1.22) or (x < 2.06 and z > -0.73 and y < 4)
            force_non_particle = (z < -1.22 and y < 3) or y > 4
        if '10-smoke' == name:
            force_particle = (x < 2 and y < 3 and z > -1.12) or (x < 1.8 and z > -0.73 and y < 4)
            force_non_particle = (z < -1.22 and y < 3) or y > 4.5 or z < -1.16 or x > 2
        if '9-smoke' == name or '11-smoke' == name or '13-smoke' == name:
            force_particle = (x < 2 and y < 3 and z > -1.12) or (x < 1.8 and z > -0.73 and y < 4) or (x > 1.9 and z > -0.4 and y < 6)
            force_non_particle = (z < -1.22 and y < 3) or y > 6.5 or z < -1.16 or x > 2 or (y > 4.5 and z < -0.6)
        if '14-smoke' == name or '15-smoke' == name:
            # Ground removal
            y1 = 1.48
            z1 = -1.28
            y2 = 12
            z2 = -0.38
            force_non_particle = z < z1 + (z2 - z1) / (y2 - y1) * (y - y1) or x > 2.8 and y < 4.4
            force_particle = x < 2.12 and 5.6 < y < 6.6 and -0.83 < z < -0.12
        if '16-smoke' == name:
            # Ground removal
            y1 = 1.48
            z1 = -1.28
            y2 = 12
            z2 = -0.38
            # force_non_particle = (z < z1 + (z2 - z1) / (y2 - y1) * (y - y1)) or (x > 2.5 and i >= 100) or (i < 100 and x > 1.5)
            force_non_particle = (z < z1 + (z2 - z1) / (y2 - y1) * (y - y1)) or x > 2.5 or (x > 1.5 and scan_id < 100) or (x > 2.34 and scan_id >= 361) or (x > 2.4 and 232 <= scan_id <= 264) or (x > 2 and z < -0.38 and 264 <= scan_id <= 300)

        if '17-smoke' == name:
            # Ground removal
            y1 = 1.48
            z1 = -1.28
            y2 = 12
            z2 = -0.38
            force_non_particle = scan_id <= 78 or (z < z1 + (z2 - z1) / (y2 - y1) * (y - y1)) or x > 2.2 or (476 <= scan_id <= 483 and z > -0.21 and x > 2)

        if '18-smoke' == name:
            # Ground removal
            y1 = 1.48
            z1 = -1.28
            y2 = 12
            z2 = -0.38
            force_non_particle = (z < z1 + (z2 - z1) / (y2 - y1) * (y - y1)) or y > 9 or x > 2

        if '19-smoke' == name:
            # Ground removal
            y1 = 1.48
            z1 = -1.28
            y2 = 12
            z2 = -0.38
            force_non_particle = (z < z1 + (z2 - z1) / (y2 - y1) * (y - y1)) or x > 2 or x < -4
            # force_particle = x < 2

        if "smoke_bush" == name:
            # Ground removal
            y1 = 1.48
            z1 = -1.23
            y2 = 9.59
            z2 = -1.25

            force_non_particle = (z < z1 + (z2 - z1) / (y2 - y1) * (y - y1)) or y > 6 or (y > 4.4 and z < -0.16 and -2.1 < x < 2) or (y > 4.9 and -2.1 < x < 2 or x < -3.8 or y> 4.6 and z > 0.83)

        if 'smoke_car' == name:
            # point 1 is the one with smallest x value because z >
            x1 = -2.11
            y1 = 4.05
            x2 = 2.05
            y2 = 3.50
            # Ground removal
            y3 = 1.49
            z3 = -1.27
            y4 = 14.7
            z4 = -0.73
            force_non_particle = (y > y1 + (y2 - y1) / (x2 - x1) * (x - x1) and -2.9 < x < x2) or (z < z3 + (z4 - z3) / (y4 - y3) * (y - y3)) or y > 6.5
            force_particle = (y < y1 + (y2 - y1) / (x2 - x1) * (x - x1) and x1 < x < x2 and z > -1.1) or (x > 2.24 and z > -1 and y < 4)

        if 'smoke_car_back' == name:
            # Ground removal
            y3 = 1.49
            z3 = -1.23
            y4 = 13.9
            z4 = -0.25
            force_non_particle = (y > 4.18 and -1.13 < x < 0.59) or z < z3 + (z4 - z3) / (y4 - y3) * (y - y3) or y > 6.5
            force_particle = (z > z3 + 0.2 + (z4 - z3) / (y4 - y3) * (y - y3) and y < 4 and x > -1.3)
        if 'smoke_car_back_far' == name:
            # Ground removal
            y3 = 1.49
            z3 = -1.23
            y4 = 13.9
            z4 = -0.25
            force_non_particle = (y >  10 and -1.13 < x < 0.59) or z < z3 + (z4 - z3) / (y4 - y3) * (y - y3) or y > 6.5
            force_particle = (z > z3 + 0.2 + (z4 - z3) / (y4 - y3) * (y - y3) and y < 4 and x < 1.69)


        if force_particle:
            if sensor == 'stereo':
                p[4] = particle_label
            else:
                p[5] = particle_label
        elif force_non_particle:
            if sensor == 'stereo':
                p[4] = 0
            else:
                p[5] = 0
    return refined_pcl

def refine_labels(name, multi=False, multi_sampled=False, lidar=False):
    num_cores = multiprocessing.cpu_count()
    # 1 load pcl
    if multi_sampled:
        print('Refining Stereo Sampled...')
        pcls = np.load(osp.join(datasets_dir, name, 'multi_pcl_sync_sampled_labelled_projected.npy'))
        pcls = Parallel(n_jobs=num_cores)(
            delayed(refine_pcl)(p, i, name, 'stereo') for i, p in tqdm.tqdm(enumerate(pcls)))
        np.save(osp.join(datasets_dir, name, 'multi_pcl_sync_sampled_labelled_projected_refined.npy'), pcls)
    if multi:
        print('Refining Stereo...')
        pcls = np.load(osp.join(datasets_dir, name, 'multi_pcl_sync_labelled_projected.npy'))
        pcls = Parallel(n_jobs=num_cores)(
            delayed(refine_pcl)(p, i, name, 'stereo') for i, p in tqdm.tqdm(enumerate(pcls)))
        np.save(osp.join(datasets_dir, name, 'multi_pcl_sync_labelled_projected_refined.npy'), pcls)
    if lidar:
        print('Refining Lidar...')
        pcls = np.load(osp.join(datasets_dir, name, 'pcl_labeled_spaces_converted_sync_cropped.npy'))
        pcls = Parallel(n_jobs=num_cores)(
            delayed(refine_pcl)(p, i, name, 'lidar') for i, p in tqdm.tqdm(enumerate(pcls)))
        np.save(osp.join(datasets_dir, name, 'pcl_labeled_spaces_converted_sync_cropped_refined.npy'), pcls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Uncomment this when deploying
    parser.add_argument('--name', type=str, help='dataset name')
    parser.add_argument('--multi', type=str, default='True', help='multi')
    parser.add_argument('--multi_sampled', type=str, default='True', help='multi_sampled')
    parser.add_argument('--lidar', type=str, default='True', help='lidar')

    args = parser.parse_args()

    args.multi = args.multi == 'True'
    args.multi_sampled = args.multi_sampled == 'True'
    args.lidar = args.lidar == 'True'

    # args.name = 'smoke_car_back_far'
    # args.multi = False
    # args.multi_sampled = True
    # args.lidar = True

    refine_labels(args.name, multi=args.multi, multi_sampled=args.multi_sampled, lidar=args.lidar)

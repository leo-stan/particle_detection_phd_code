import numpy as np
import os.path as osp
import sys
sys.path.append('../')
from particle_detection.src.config import cfg
import argparse
# Load data
# Input: a pointcloud numpy file from julian labeling (name_labeled or name_labeled_spaces)

# filename = [
#     'smoke_bush_stereo'
# ]

def convert_julian_pcl(f):
    print("Converting file %s..." % f)

    pcl = np.load(osp.join(cfg.RAW_DATA_DIR, f, 'pcl_labeled_spaces.npy'))
    # pcl = np.load('/home/leo/phd/particle_detection/src/particle_detection/data/2-dust/pcl_labeled_spaces.npy')
    # pcl_new = []
    # for p in pcl:
    #     pcl_new.append(np.concatenate((p[:, :3],p[:, 4].reshape(-1, 1), np.zeros((p.shape[0], 2))), axis=1))
    # #
    # # np.save(osp.join(cfg.RAW_DATA_DIR,f+'_converted.npy'),pcl_new)
    # np.save('/home/leo/phd/particle_detection/src/particle_detection/data/2-dust/pcl_labeled_spaces_converted.npy',pcl_new)
    # print('Conversion finished')

    pcl_new = []
    for p in pcl:
        p_diff = p[np.sum(p[:, :3] - p[:, 5:8], axis=1) != 0, :]  # Find echo1 points that differ from echo2
        labels_echo1 = p_diff[:, -3:]
        labels_echo1 = np.sum(labels_echo1 * np.array([0, 1, 2]), axis=1)
        p_echo1 = np.concatenate((p_diff[:, 5:9], 1 * np.ones((p_diff.shape[0], 1)), labels_echo1.reshape(-1, 1)), axis=1)

        labels_echo2 = np.zeros((p_diff.shape[0], 1)) # second echo is never particle
        p_echo2 = np.concatenate((p_diff[:, :4], 2 * np.ones((p_diff.shape[0], 1)), labels_echo2),
                                 axis=1)  # Add 2 for second echo and label=0 because assumed non-particle

        p_both = p[np.sum(p[:, :3] - p[:, 5:8], axis=1) == 0, :]
        labels_both = p_both[:, -3:]
        labels_both = np.sum(labels_both * np.array([0, 1, 2]), axis=1)
        p_both = np.concatenate((p_both[:, :4], np.zeros((p_both.shape[0], 1)), labels_both.reshape(-1, 1)), axis=1)
        pcl_new.append(np.concatenate((p_both, p_echo1, p_echo2)))

    np.save(osp.join(cfg.RAW_DATA_DIR, f, 'pcl_labeled_spaces_converted.npy'), pcl_new)

    print('Conversion finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Uncomment this when deploying
    parser.add_argument('--name', type=str, help='dataset name')

    args = parser.parse_args()

    print('convert julian pcl dataset %s' % args.name)

    # for dataset in datasets:
    convert_julian_pcl(args.name)
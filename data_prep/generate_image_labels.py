import os
import os.path as osp
import sys

import cv2
import numpy as np
import tqdm

sys.path.insert(0, '../')
from particle_detection.src.config import cfg
import argparse


# Arguments:
# datasets = [
# '1-dust',
# '2-dust',
# '3-dust',
# '4-dust',
# '5-dust',
# '6-dust',
# '7-dust',
# '8-dust',
# '9-smoke',
# '10-smoke',
# '11-smoke',
# '12-smoke',
# '13-smoke',
# '14-smoke',
# '15-smoke',
# '16-smoke',
# '17-smoke',
# '18-smoke',
# '19-smoke',
# 'smoke_bush'
# ]
# augment_imgs = False

def generate_png_imgs(dataset):
    imgs = np.load(osp.join(cfg.RAW_DATA_DIR, dataset, 'images_sync.npy'))
    rgbs = np.load(osp.join(cfg.RAW_DATA_DIR, dataset, 'images_sync_rgb.npy'))
    imgs_depth = np.load(osp.join(cfg.RAW_DATA_DIR, dataset, 'images_sync_depth.npy'))
    imgs_label = np.load(osp.join(cfg.RAW_DATA_DIR, dataset, 'images_sync_label.npy'))

    # if not augment_imgs:
    image_dir = osp.join(cfg.RAW_DATA_DIR, dataset, 'images')
    rgb_dir = osp.join(cfg.RAW_DATA_DIR, dataset, 'images_rgb')
    label_dir = osp.join(cfg.RAW_DATA_DIR, dataset, 'images_label')
    depth_dir = osp.join(cfg.RAW_DATA_DIR, dataset, 'images_depth')
    # else:
    #     image_dir = osp.join(cfg.RAW_DATA_DIR, dataset, 'aug_images')
    #     label_dir = osp.join(cfg.RAW_DATA_DIR, dataset, 'aug_image_labels')
    #     depth_dir = osp.join(cfg.RAW_DATA_DIR, dataset, 'aug_image_depth')

    if not osp.exists(image_dir):
        os.makedirs(image_dir)
    else:
        # clean directory
        for f in os.listdir(image_dir):
            os.remove(osp.join(image_dir, f))

    if not osp.exists(rgb_dir):
        os.makedirs(rgb_dir)
    else:
        # clean directory
        for f in os.listdir(rgb_dir):
            os.remove(osp.join(rgb_dir, f))

    if not osp.exists(label_dir):
        os.makedirs(label_dir)
    else:
        # Clean directory
        for f in os.listdir(label_dir):
            os.remove(osp.join(label_dir, f))

    if not osp.exists(depth_dir):
        os.makedirs(depth_dir)
    else:
        # Clean directory
        for f in os.listdir(depth_dir):
            os.remove(osp.join(depth_dir, f))

    for i, (img, rgb, img_label, img_depth) in tqdm.tqdm(enumerate(zip(imgs, rgbs, imgs_label, imgs_depth))):
        cv2.imwrite(osp.join(image_dir, str(i) + '.png'), img)
        cv2.imwrite(osp.join(rgb_dir, str(i) + '.png'), rgb)
        cv2.imwrite(osp.join(label_dir, str(i) + '.png'), img_label)
        # cv2.imwrite(osp.join(depth_dir, str(i)conf + '.png'), img_depth)
        np.save(osp.join(depth_dir, str(i) + '.npy'), img_depth)
    # if augment_imgs:
    #     print("Augmenting images...")
    #     # Add augmented images
    #     for img, label in zip(imgs,labels):
    #         cv2.imwrite(osp.join(image_dir,str(i) + '.png'), np.flipud(img))
    #         cv2.imwrite(osp.join(label_dir,str(i) + '.png'), np.flipud(label))
    #         i += 1
    #     for img, label in zip(imgs,labels):
    #         cv2.imwrite(osp.join(image_dir,str(i) + '.png'), np.fliplr(img))
    #         cv2.imwrite(osp.join(label_dir,str(i) + '.png'), np.fliplr(label))
    #         i += 1
    #     for img, label in zip(imgs,labels):
    #         cv2.imwrite(osp.join(image_dir,str(i) + '.png'), np.fliplr(np.flipud(img)))
    #         cv2.imwrite(osp.join(label_dir,str(i) + '.png'), np.fliplr(np.flipud(label)))
    #         i += 1

    # print("Saving images in numpy array")
    # np.save(numpy_save, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--name', type=str, help='dataset name')

    args = parser.parse_args()

    # args.name = '12-smoke'

    generate_png_imgs(args.name)

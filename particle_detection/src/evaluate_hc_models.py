#!/usr/bin/env python

import sys

sys.path.insert(0, '../..')

import os
import os.path as osp
import numpy as np
import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import yaml
from particle_dataset import ParticleDataset
from particle_detection.utils.segment_scan import segment_scan
from config import cfg
from datetime import datetime
from sklearn.externals import joblib
from torch.utils.data import DataLoader
import argparse
import time

def detection_collate(batch):
    voxel_features = []
    voxel_coords = []
    labels = []

    for i, sample in enumerate(batch):
        voxel_features.append(sample['inputs'])

        voxel_coords.append(
            np.pad(sample['coords'], ((0, 0), (1, 0)),
                   mode='constant', constant_values=i))

        labels.append(sample['labels'])

    return {
        'inputs': np.concatenate(voxel_features),
        'coords': np.concatenate(voxel_coords),
        'labels': np.concatenate(labels)}


def evaluate_model(input_model, eval_title=None, test_data=None, save_pcl=True):
    datasets_dir = osp.join(cfg.DATASETS_DIR, 'hc_datasets')
    logs_dir = osp.join(cfg.LOG_DIR, 'hc_logs')

    # Load dataset parameters
    with open(osp.join(logs_dir, input_model, 'config.yaml'), 'r') as f:
        log_params = yaml.load(f, Loader=yaml.SafeLoader)

    # Evaluation parameters
    if not eval_title:
        eval_title = 'testing'
    if not test_data:
        test_data = log_params['train_data'].replace('training', 'testing')

    now = datetime.now()

    # Load dataset parameters
    with open(osp.join(datasets_dir, test_data, 'config.yaml'), 'r') as f:
        ds_parameters = yaml.load(f, Loader=yaml.SafeLoader)
    nb_scans = ds_parameters['nb_scans']

    # Load model parameters
    with open(osp.join(logs_dir, input_model, 'config.yaml'), 'r') as f:
        parameters = yaml.load(f, Loader=yaml.SafeLoader)

    eval_dir = osp.join(logs_dir, input_model, 'evaluations/eval_' + now.strftime('%Y%m%d_%H%M%S') + '_' + eval_title)
    os.makedirs(eval_dir)

    # Load model
    model = joblib.load(osp.join(logs_dir, input_model, 'model.pkl'))

    # Load scaler
    scaler = joblib.load(osp.join(logs_dir, input_model, 'scaler.pkl'))

    test_loader = DataLoader(ParticleDataset(dataset_dir=osp.join(datasets_dir, test_data), scaler=scaler),
                             batch_size=1, shuffle=False, num_workers=4, collate_fn=detection_collate)

    eval_pcl = []
    y_pred = []
    y_target = []

    for s, sample in tqdm.tqdm(enumerate(test_loader), total=len(test_loader),
                               desc='Evaluating scan', ncols=80,
                               leave=False):

        pcl = np.load(osp.join(datasets_dir, test_data, 'scan_pcls', str(s) + '.npy'))

        selected_inputs = np.array([]).reshape(sample['inputs'].shape[0], 0)

        if 'roughness' in parameters['features']:
            selected_inputs = np.concatenate((selected_inputs, sample['inputs'][:, 0].reshape(-1, 1)), axis=1)
        if 'slope' in parameters['features']:
            selected_inputs = np.concatenate((selected_inputs, sample['inputs'][:, 1].reshape(-1, 1)), axis=1)
        if 'intmean' in parameters['features']:
            selected_inputs = np.concatenate((selected_inputs, sample['inputs'][:, 2].reshape(-1, 1)), axis=1)
        if 'intmean' in parameters['features']:
            selected_inputs = np.concatenate((selected_inputs, sample['inputs'][:, 3].reshape(-1, 1)), axis=1)
        if 'echo' in parameters['features']:
            selected_inputs = np.concatenate((selected_inputs, sample['inputs'][:, 4:7]), axis=1)

        # start = time.time()
        pred = model.predict(selected_inputs)
        # end = time.time()
        # print(end-start)
        raw_pred_points = segment_scan(pcl, pred, ds_parameters['map_config'])

        # Removes any scan with ground truth = 1 for whole scan
        if np.sum((raw_pred_points[:, 5] == 1).astype(np.int8)) / float(raw_pred_points.shape[0]) < 0.5:
            if save_pcl:
                eval_pcl.append(raw_pred_points[:, :7])
            y_pred.append(raw_pred_points[:, 6].astype(np.uint8))
            y_target.append(raw_pred_points[:, 5].astype(np.uint8))

    if save_pcl:
        np.save(osp.join(eval_dir, 'scans'), eval_pcl)
    y_pred = np.concatenate(y_pred).astype(np.uint8)
    y_target = np.concatenate(y_target).astype(np.uint8)

    with open(osp.join(eval_dir, 'eval_results.txt'), 'w') as f:

        f.write('Evaluation parameters:\n')
        f.write('ParticleDataset: %s\n' % test_data)
        f.write('nb_scans: %s\n' % ds_parameters['nb_scans'])
        f.write('dataset_size: %s\n' % ds_parameters['dataset_size'])
        f.write('\n\nEvaluation results:\n')

        # Compute performance scores
        print('\n')
        print("Evaluation results for model: %s" % input_model)

        print('Confusion Matrix')
        f.write("Confusion Matrix\n")

        cnf_matrix = confusion_matrix(y_target, y_pred).astype(np.float64)  # Compute confusion matrix
        cnf_matrix /= cnf_matrix.sum(1, keepdims=True)  # put into ratio
        cnf_matrix = cnf_matrix[:2, :2]
        np.set_printoptions(precision=2)
        print(cnf_matrix)
        f.write(str(cnf_matrix))
        f.write('\n')

        # Can only use this if both classes are at least predicted once
        if len(np.unique(y_pred)) > 1:
            print('Classification Report')
            f.write('Classification Report\n')

            cr = classification_report(y_target, y_pred, digits=2)
            print(cr)
            f.write(cr)
            f.write('\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--name', type=str, help='log name')

    args = parser.parse_args()

    # Models
    # input_models = [
        # '20190615_184107_both_int_relpos_echo',
        # '20190616_160218_small_smoke_dust_all'
        # '20190619_183149_smallest_smoke_dust_all'
        # '20190619_183402_smallest_smoke_dust_all'
        # '20190617_122516_RF_small_smoke_dust_all'
        # '20190718_205330_naive_bayes_smoke_dust_intmean_intvar',
        # '20190718_205345_naive_bayes_smoke_dust_roughness_slope',
        # '20190718_205409_naive_bayes_smoke_dust_echo',
        # '20190718_205434_naive_bayes_smoke_dust_intmean_intvar_roughness_slope',
        # '20190718_205458_naive_bayes_smoke_dust_intmean_intvar_echo',
        # '20190718_205525_naive_bayes_smoke_dust_roughness_slope_echo',
        # '20190718_205552_naive_bayes_smoke_dust_intmean_intvar_roughness_slope_echo',
        # '20190617_153329_naive_bayes_small_smoke_dust_all'
    # ]
    # for m in input_models:
    evaluate_model(args.name, test_data='visualisation_smoke_dust_intmean_intvar_roughness_slope_echo', save_pcl=True)

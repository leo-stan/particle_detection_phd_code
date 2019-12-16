#!/usr/bin/env python

import sys

sys.path.insert(0, '../../')

import os
import os.path as osp
import numpy as np
import torch
import tqdm
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, brier_score_loss, precision_recall_curve, precision_score, recall_score, f1_score
import yaml
import particle_detection.models
from particle_dataset import ParticleDataset, ParticleFusionDataset
from particle_detection.utils.segment_scan import segment_scan
from config import cfg
from datetime import datetime
from sklearn.externals import joblib
from torch.utils.data import DataLoader
from torch.autograd import Variable
import math
import argparse
import time
from particle_detection.utils.asl_score import average_score_loss
import csv

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


def evaluate_model(sensor, input_model, eval_title=None, set='testing', test_data=None, save_pcl=False, save_voxel=False, proba_pred=True,
                   pr_curve=True, dos=0, voxel_eval=True,missing_data=False, match_only=True):
    if sensor == 'lidar':
        datasets_dir = osp.join(cfg.DATASETS_DIR, 'dl_datasets')
        logs_dir = osp.join(cfg.LOG_DIR, 'dl_logs')
    elif sensor == 'stereo':
        datasets_dir = osp.join(cfg.DATASETS_DIR, 'st_datasets')
        logs_dir = osp.join(cfg.LOG_DIR, 'st_logs')
    else:
        datasets_dir = []
        datasets_dir.append(osp.join(cfg.DATASETS_DIR, 'dl_datasets'))
        datasets_dir.append(osp.join(cfg.DATASETS_DIR, 'st_datasets'))
        logs_dir = osp.join(cfg.LOG_DIR, 'fusion_logs')
        # Force voxel eval when doing fusion
        voxel_eval = True
        save_pcl = False

    # If list of models provided then treat evaluation as ensemble
    if isinstance(input_model, list):
        ensemble_eval = True
        # Load dataset parameters
        with open(osp.join(logs_dir, input_model[0], 'config.yaml'), 'r') as f:
            log_params = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        ensemble_eval = False
        # Load dataset parameters
        with open(osp.join(logs_dir, input_model, 'config.yaml'), 'r') as f:
            log_params = yaml.load(f, Loader=yaml.SafeLoader)

    # Evaluation parameters
    if not eval_title:
        eval_title = set
    if voxel_eval:
        eval_title += '_voxel'

    # If test data is not provided then use default (same name but testing)
    if not test_data:
        test_data = log_params['train_data'].replace('training', set)
        if ensemble_eval:
            # remove digit number from training set name
            test_data = test_data[:-1]

    now = datetime.now()

    # Load dataset parameters
    if 'fusion' in sensor:
        with open(osp.join(datasets_dir[0], test_data[0], 'config.yaml'), 'r') as f:
            dataset_params = yaml.load(f, Loader=yaml.SafeLoader)
        with open(osp.join(datasets_dir[1], test_data[1], 'config.yaml'), 'r') as f:
            stereo_dataset_params = yaml.load(f, Loader=yaml.SafeLoader)

        assert dataset_params['map_config'] == stereo_dataset_params['map_config'], 'map config mismatch'
        map_config = dataset_params['map_config']
        nb_scans = dataset_params['nb_scans']

    else:
        with open(osp.join(datasets_dir, test_data, 'config.yaml'), 'r') as f:
            dataset_params = yaml.load(f, Loader=yaml.SafeLoader)

            map_config = dataset_params['map_config']
            nb_scans = dataset_params['nb_scans']

    # Choose device for computation
    # if torch.cuda.is_available():
    #     device = torch.device("cuda:1")
    # else:
    #     device = torch.device("cpu")

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cuda = torch.cuda.is_available()

    if ensemble_eval:
        model = []
        scaler = []
        for m in input_model:
            # Load model parameters
            with open(osp.join(logs_dir, m, 'config.yaml'), 'r') as f:
                parameters = yaml.load(f, Loader=yaml.SafeLoader)

            if parameters['conv']:
                if sensor == 'fusion_mid':
                    model.append(particle_detection.models.SmokeNet3DConvMid(model_config=parameters['model_config'],
                                                                             map_config=map_config,
                                                                             batch_size=1))
                else:
                    model.append(particle_detection.models.SmokeNet3DConv(model_config=parameters["model_config"],
                                                                          map_config=map_config, batch_size=1))
            else:
                model.append(particle_detection.models.SmokeNetNoConv(model_config=parameters["model_config"],
                                                                      map_config=map_config))
            # Load the best model saved
            model_state = osp.join(logs_dir, m, 'best.pth.tar')
            model_state = torch.load(model_state)
            model[-1].load_state_dict(model_state['model_state_dict'])
            model[-1].eval()
            model[-1].cuda()
            if 'fusion' not in sensor:
                scaler.append(joblib.load(osp.join(logs_dir, m, 'scaler.pkl')))
            else:
                scaler.append([joblib.load(osp.join(logs_dir, m, 'lidar_scaler.pkl')), joblib.load(osp.join(logs_dir, m, 'stereo_scaler.pkl'))])
        eval_dir = osp.join(logs_dir, 'ensemble_evaluations/eval_' + now.strftime('%Y%m%d_%H%M%S') + '_' + eval_title)
        os.makedirs(eval_dir)
    else:
        # Load model parameters
        with open(osp.join(logs_dir, input_model, 'config.yaml'), 'r') as f:
            parameters = yaml.load(f, Loader=yaml.SafeLoader)

        eval_dir = osp.join(logs_dir, input_model, 'evaluations/eval_' + now.strftime('%Y%m%d_%H%M%S') + '_' + eval_title)
        os.makedirs(eval_dir)

        if parameters['conv']:
            if parameters['dos']:
                model = particle_detection.models.SmokeNet3DConvDOS(model_config=parameters["model_config"],
                                                                    map_config=map_config)
            else:
                if sensor == 'fusion_mid':
                    model = particle_detection.models.SmokeNet3DConvMid(model_config=parameters['model_config'],
                                                                        map_config=map_config,
                                                                        batch_size=1)
                else:
                    model = particle_detection.models.SmokeNet3DConv(model_config=parameters["model_config"],
                                                                     map_config=map_config, batch_size=1)
        else:
            model = particle_detection.models.SmokeNetNoConv(model_config=parameters["model_config"], map_config=map_config)
        # Load the best model saved
        model_state = osp.join(logs_dir, input_model, 'best.pth.tar')
        model_state = torch.load(model_state)
        model.load_state_dict(model_state['model_state_dict'])
        model.eval()
        model.cuda()
        if 'fusion' not in sensor:
            scaler = joblib.load(osp.join(logs_dir, input_model, 'scaler.pkl'))
        else:
            scaler = [joblib.load(osp.join(logs_dir, input_model, 'lidar_scaler.pkl')), joblib.load(osp.join(logs_dir, input_model, 'stereo_scaler.pkl'))]


    y_target = []
    y_pred = []
    y_mean_vec = []
    y_std_vec = []
    val_loss = 0

    eval_pcl = []
    eval_voxel = []

    # Load data without scaling if ensemble evaluation
    if 'fusion' not in sensor:
        if ensemble_eval:
            test_loader = DataLoader(ParticleDataset(dataset_dir=osp.join(datasets_dir, test_data)),
                                     batch_size=1, shuffle=False, num_workers=4, collate_fn=detection_collate)
        else:
            test_loader = DataLoader(ParticleDataset(dataset_dir=osp.join(datasets_dir, test_data), scaler=scaler),
                                 batch_size=1, shuffle=False, num_workers=4, collate_fn=detection_collate)
    else:
        if ensemble_eval:
            test_loader = DataLoader(
                ParticleFusionDataset(lidar_dataset_dir=osp.join(datasets_dir[0], test_data[0]),
                                      stereo_dataset_dir=osp.join(datasets_dir[1], test_data[1]), match_only=match_only),
                batch_size=1, shuffle=False, num_workers=4,
                collate_fn=detection_collate)
        else:
            test_loader = DataLoader(
                ParticleFusionDataset(lidar_dataset_dir=osp.join(datasets_dir[0], test_data[0]), stereo_dataset_dir=osp.join(datasets_dir[1], test_data[1]),lidar_scaler=scaler[0],stereo_scaler=scaler[1], match_only=match_only),
                batch_size=1, shuffle=False, num_workers=4,
                collate_fn=detection_collate)

    sm = nn.Softmax(dim=1)
    pred_time = 0
    for s, sample in tqdm.tqdm(enumerate(test_loader), total=len(test_loader),
                               desc='Evaluating scan', ncols=80,
                               leave=False):
        # Removes any scan with ground truth = 1 for whole scan (deprecated, should not happen anymore)
        if np.sum((sample['labels'] == 1).astype(np.int8)) / float(sample['labels'].shape[0]) < 0.5:

            if not ensemble_eval:
                inputs = Variable(torch.FloatTensor(sample['inputs']))
                inputs = inputs.cuda()

            if parameters["conv"]:
                coords = Variable(torch.LongTensor(sample['coords']))
                coords = coords.cuda()

                # MC dropout evaluation
                if parameters["dos"] and dos > 0 and not ensemble_eval:
                    model.train()
                    for i in range(dos):
                        if i == 0:
                            with torch.no_grad():
                                pred = sm(model(inputs, coords))

                            pred = pred.unsqueeze(0)
                        else:
                            with torch.no_grad():
                                pred = torch.cat((pred, sm(model(inputs, coords)).unsqueeze(0)), dim=0)
                else:
                    # Ensemble evaluation
                    if ensemble_eval:

                        for i in range(len(model)):

                            # Load the input and scale for each model
                            if missing_data:
                                inputs = np.zeros(sample['inputs'].shape)
                            else:
                                inputs = sample['inputs'].copy()

                            if 'fusion' not in sensor:
                                inputs = scaler[i].transform(inputs.reshape(-1, inputs.shape[2])).reshape(-1,
                                                                                                      inputs.shape[1],
                                                                                                      inputs.shape[2])
                            else:
                                # Scale the lidar and stereo data separately and concat
                                inputs = np.concatenate((scaler[i][0].transform(inputs[:, :, :7].reshape(-1, 7)).reshape(-1,
                                                                                                      inputs.shape[1],
                                                                                                      7),scaler[i][1].transform(inputs[:, :, 7:].reshape(-1, 6)).reshape(-1,
                                                                                                      inputs.shape[1],
                                                                                                      6)),axis=2)
                            inputs = Variable(torch.FloatTensor(inputs))
                            inputs = inputs.cuda()
                            start = time.time()
                            if i == 0:
                                pred = sm(model[i](inputs, coords))
                                pred = pred.unsqueeze(0)
                            else:
                                pred = torch.cat((pred, sm(model[i](inputs, coords)).unsqueeze(0)), dim=0)
                            end = time.time()
                            pred_time += end - start
                    # Single model evaluation
                    else:
                        with torch.no_grad():
                            # start = time.time()
                            pred = sm(model(inputs, coords))
                            # end = time.time()
                            # pred_time += end - start
            else:
                with torch.no_grad():
                    pred = sm(model(inputs))

            # Compute mean and std from MC dropout or ensemble prediction
            if (parameters["dos"] and dos > 0) or ensemble_eval:
                proba_mean = pred.mean(dim=0)
                proba_std = pred.std(dim=0)[:, 1].cpu().data.numpy()
                pred = proba_mean.argmax(dim=1).cpu().data.numpy()
                proba_mean = proba_mean[:, 1].cpu().data.numpy()

                # compute raw pcl prediction from voxel prediction if pcl eval or need to save pcl
                if not voxel_eval or save_pcl:
                    pcl = np.load(osp.join(datasets_dir, test_data, 'scan_pcls', str(s) + '.npy'))
                    if proba_pred:
                        raw_pred_points = segment_scan(pcl, pred, map_config, proba_mean, proba_std)
                    else:
                        raw_pred_points = segment_scan(pcl, pred, map_config)
                # np.save(osp.join(eval_dir, 'scans', str(s)), raw_pred_points[:, :7])
                    if save_pcl:
                        eval_pcl.append(raw_pred_points)
                if save_voxel:
                    if proba_pred:
                        eval_voxel.append(
                            np.concatenate((sample['coords'][:, 1:] * np.array((map_config[
                                                                                    'voxel_size_x'],
                                                                                map_config[
                                                                                    'voxel_size_y'],
                                                                                map_config[
                                                                                    'voxel_size_z'])),
                                            sample['labels'].reshape(-1, 1),
                                            pred.reshape(-1, 1), proba_mean.reshape(-1, 1),proba_std.reshape(-1, 1)), axis=1))
                    else:
                        eval_voxel.append(
                            np.concatenate(
                                (sample['coords'][:, 1:], sample['labels'].reshape(-1, 1), pred.reshape(-1, 1)),
                                axis=1))

            # Compute prediction for single model
            else:

                [_, index] = pred.max(dim=1)

                if proba_pred:
                    proba_mean = pred[:, 1].cpu().data.numpy()

                pred = index.cpu().data.numpy()

                # compute raw pcl prediction from voxel prediction if pcl eval or need to save pcl
                if not voxel_eval or save_pcl:
                    pcl = np.load(osp.join(datasets_dir, test_data, 'scan_pcls', str(s) + '.npy'))
                    if proba_pred:
                        raw_pred_points = segment_scan(pcl, pred, map_config, output_proba_mean=proba_mean)
                    else:
                        raw_pred_points = segment_scan(pcl, pred, map_config)

                    if save_pcl:
                        if sensor == 'lidar':
                            eval_pcl.append(raw_pred_points[:, :8])
                        else:
                            eval_pcl.append(raw_pred_points[:, :7])

                if save_voxel:
                    if proba_pred:
                        eval_voxel.append(np.concatenate((sample['coords'][:, 1:] * np.array((map_config[
                                                                                                  'voxel_size_x'],
                                                                                              map_config[
                                                                                                  'voxel_size_y'],
                                                                                              map_config[
                                                                                                  'voxel_size_z'])),
                                                          sample['labels'].reshape(-1, 1),
                                                          pred.reshape(-1, 1), proba_mean.reshape(-1, 1)), axis=1))
                    else:
                        eval_voxel.append(
                            np.concatenate((sample['coords'][:, 1:], sample['labels'].reshape(-1, 1), pred.reshape(-1, 1)),
                                           axis=1))

            if voxel_eval:
                if proba_pred:
                    y_mean_vec.append(proba_mean)
                    if (parameters["dos"] and dos > 0) or ensemble_eval:
                        y_std_vec.append(proba_std)
                y_pred.append(pred)
                y_target.append(sample['labels'])
            else:
                if proba_pred:
                    if sensor == 'lidar':
                        y_mean_vec.append(raw_pred_points[:, 7])
                        if (parameters["dos"] and dos > 0) or ensemble_eval:
                            y_std_vec.append(raw_pred_points[:, 8])
                    else:
                        y_mean_vec.append(raw_pred_points[:, 6])
                        if (parameters["dos"] and dos > 0) or ensemble_eval:
                            y_std_vec.append(raw_pred_points[:, 7])
                if sensor == 'lidar':
                    y_pred.append(raw_pred_points[:, 6].astype(np.uint8))
                    y_target.append(raw_pred_points[:, 5].astype(np.uint8))
                else:
                    y_pred.append(raw_pred_points[:, 5].astype(np.uint8))
                    y_target.append(raw_pred_points[:, 4].astype(np.uint8))
    if save_pcl:
        np.save(osp.join(eval_dir, 'scans.npy'), eval_pcl)
    if save_voxel:
        np.save(osp.join(eval_dir, 'scans_voxel.npy'), eval_voxel)
    y_pred = np.concatenate(y_pred).astype(np.uint8)
    if proba_pred:
        y_mean_vec = np.concatenate(y_mean_vec)
        if (parameters["dos"] and dos > 0) or ensemble_eval:
            y_std_vec = np.concatenate(y_std_vec)
    y_target = np.concatenate(y_target).astype(np.uint8)

    y_mean_vec = y_mean_vec[y_target != 2]
    y_std_vec = y_std_vec[y_target != 2]
    y_pred = y_pred[y_target != 2]
    y_target = y_target[y_target != 2]

    pred_time /= float(nb_scans)

    with open(osp.join(eval_dir, 'eval_results.csv'), 'a') as results:
        result_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        result_writer.writerow(
            ['precision', 'recall', 'f1', 'BSL','ASL'])

    with open(osp.join(eval_dir, 'eval_results.txt'), 'w') as f:

        f.write('Evaluation parameters:\n')
        f.write('ParticleDataset: %s\n' % test_data)
        f.write('nb_scans: %s\n' % nb_scans)
        f.write('avg_pred_time: %f\n' % pred_time)
        f.write('Voxel Eval: %r\n' % voxel_eval)
        f.write('Dropout Sampling: %d\n' % dos)
        f.write('Matching points only: %r\n' % match_only)
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

            cr = classification_report(y_target, y_pred, digits=3)
            print(cr)
            f.write(cr)
            f.write('\n')

        # Brier score
        bsl = brier_score_loss(y_target, y_mean_vec)
        print('Brier Score Loss: %f' % bsl)
        f.write('Brier Score Loss: %f\n' % bsl)

        asl = average_score_loss(y_target, y_mean_vec)
        print('Average Score Loss: %f' % asl)
        f.write('Average Score Loss: %f\n' % asl)

        # Negative Log-Likelihood
        # nll = 0
        # for i in range(y_target.shape[0]):
        #     if y_target[i].astype(int) == 1:
        #         nll += -math.log(max(0.001, y_mean_vec[i]))
        #     else:
        #         nll += -math.log(max(0.001, 1 - y_mean_vec[i]))
        # nll /= y_target.shape[0]
        # print('Negative Log Likelihood: %f' % nll)
        # f.write('Negative Log Likelihood: %f\n' % nll)

        if pr_curve and proba_pred:
            precision, recall, thresholds = precision_recall_curve(y_target, y_mean_vec)
            np.save(osp.join(eval_dir, 'PR_curve.npy'), [precision, recall])

            F1 = 2 * precision * recall / (precision + recall)

            best_id = np.argmax(F1)
            best_t = thresholds[best_id]
            best_f1 = F1[best_id]
            best_p = precision[best_id]
            best_r = recall[best_id]

            f.write('best_t: %f\n' % best_t)
            f.write('best_f1: %f\n' % best_f1)
            f.write('best_p: %f\n' % best_p)
            f.write('best_r: %f\n' % best_r)

            # import matplotlib.pyplot as plt
            # plt.figure(1)
            # plt.plot(recall, precision, '.', label='PR Curve')
            # plt.xlabel('Recall')
            # plt.ylabel('Precision')
            # plt.ylim([0.0, 1.05])
            # plt.xlim([0.0, 1.0])
            # plt.grid(True)
            # plt.title('Precision recall curve')
            # plt.legend(loc='lower left')
            # plt.savefig(osp.join(eval_dir, 'PRcurve.png'))
            # plt.show()

    with open(osp.join(eval_dir, 'eval_results.csv'), 'a') as results:
        result_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        result_writer.writerow(
            [precision_score(y_target, y_pred), recall_score(y_target, y_pred), f1_score(y_target, y_pred), bsl,
             asl])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--name', type=str, help='log name')
    parser.add_argument('--eval_title', type=str, help='eval_title')
    parser.add_argument('--set', type=str, help='e.g. "testing"')
    parser.add_argument('--sensor', type=str, help='sensor')
    parser.add_argument('--save_pcl', type=str, default='False', help='save predicted pcl flag')
    parser.add_argument('--save_voxel', type=str, default='False', help='save predicted voxel flag')
    parser.add_argument('--voxel_eval', type=str, default='False', help='voxel eval flag')
    parser.add_argument('--nb_en', type=int, default=5, help='nb_ensembles')
    parser.add_argument('--missing_data', action='store_true', help='missing data flag')
    parser.add_argument('--close', action='store_true', help='close flag')

    args = parser.parse_args()
    args.save_pcl = args.save_pcl == 'True'
    args.save_voxel = args.save_voxel == 'True'
    args.voxel_eval = args.voxel_eval == 'True'

    if args.set == 'visualisation':
        args.save_pcl = True
        args.save_voxel = True
    # sensor = 'lidar'
    # sensor = 'stereo'

    # eval_title = None
    test_data = None
    # eval_title = '12-smoke'
    # test_data = 'visualisation_smoke_dust_relpos_rgb'
    # args.name = '20190809_121706_conv_smoke_dust_relpos_rgb_real1'
    # args.sensor = 'stereo'

    # test_data = 'visualisation_smoke_dust_relpos_rgb'
    # args.name = '20190720_222224_conv_smoke_dust_int_relpos_echo'
    # args.sensor = 'lidar'

    # eval_title = 'test'
    # args.sensor = 'lidar'
    # test_data = 'visualisation_smoke_dust_int_relpos_echo'
    # args.name = [
    #     '20190701_140416_smoke_dust_int_relpos_echo',
    #     '20190720_222224_conv_smoke_dust_int_relpos_echo',
    #     # '20190701_140416_smoke_dust_int_relpos_echo',
    #     # '20190720_222224_conv_smoke_dust_int_relpos_echo',
    #     # '20190701_140416_smoke_dust_int_relpos_echo',
    # ]

    # HPC training
    if args.sensor == 'stereo':
        args.name = [
            '20190905_192925_conv_smoke_dust_relpos_rgb_fusion00',
            '20190905_193146_conv_smoke_dust_relpos_rgb_fusion44',
            '20190905_193151_conv_smoke_dust_relpos_rgb_fusion11',
            '20190905_193154_conv_smoke_dust_relpos_rgb_fusion22',
            '20190905_193154_conv_smoke_dust_relpos_rgb_fusion33',
            '20190906_202453_conv_smoke_dust_relpos_rgb_fusion00',
            '20190906_202453_conv_smoke_dust_relpos_rgb_fusion11',
            '20190906_223852_conv_smoke_dust_relpos_rgb_fusion22',
            '20190906_225859_conv_smoke_dust_relpos_rgb_fusion44',
            '20190906_225900_conv_smoke_dust_relpos_rgb_fusion33',
        ]
        test_data = args.set + '_smoke_dust_relpos_rgb_fusion'
        if args.close:
            args.name = [
                '20190908_024535_conv_smoke_dust_relpos_rgb_fusion_close00',
                '20190908_024911_conv_smoke_dust_relpos_rgb_fusion_close11',
                '20190908_042044_conv_smoke_dust_relpos_rgb_fusion_close22',
                '20190908_042202_conv_smoke_dust_relpos_rgb_fusion_close33',
                '20190908_050957_conv_smoke_dust_relpos_rgb_fusion_close44',
                '20190908_174724_conv_smoke_dust_relpos_rgb_fusion_close33',
                '20190908_174724_conv_smoke_dust_relpos_rgb_fusion_close44',
                '20190908_174727_conv_smoke_dust_relpos_rgb_fusion_close00',
                '20190908_174735_conv_smoke_dust_relpos_rgb_fusion_close11',
                '20190908_174735_conv_smoke_dust_relpos_rgb_fusion_close22',
            ]
            test_data += '_close_40m'

    elif args.sensor == 'lidar':
        args.name = [
            '20190908_210535_conv_smoke_dust_int_relpos_echo_fusion00',
            '20190908_210535_conv_smoke_dust_int_relpos_echo_fusion11',
            '20190908_210536_conv_smoke_dust_int_relpos_echo_fusion22',
            '20190909_001845_conv_smoke_dust_int_relpos_echo_fusion33',
            '20190909_002319_conv_smoke_dust_int_relpos_echo_fusion44',
            '20190909_003144_conv_smoke_dust_int_relpos_echo_fusion00',
            '20190909_013203_conv_smoke_dust_int_relpos_echo_fusion11',
            '20190909_032834_conv_smoke_dust_int_relpos_echo_fusion22',
            '20190909_033533_conv_smoke_dust_int_relpos_echo_fusion33',
            '20190909_034047_conv_smoke_dust_int_relpos_echo_fusion44',
        ]
        test_data = args.set + '_smoke_dust_int_relpos_echo_fusion'
        if args.close:
            test_data += '_close_40m'

    elif args.sensor == 'fusion':
        args.name = [
            # # '20190915_125042_conv_smoke_dust_relpos_rgb22',
            # '20190915_125043_conv_smoke_dust_relpos_rgb00',
            # # '20190915_125048_conv_smoke_dust_relpos_rgb11',
            # '20190915_135600_conv_smoke_dust_relpos_rgb_fusion33',
            # '20190915_144244_conv_smoke_dust_relpos_rgb_fusion44',
            # '20190915_144417_conv_smoke_dust_relpos_rgb_fusion00',
            # '20190915_155048_conv_smoke_dust_relpos_rgb_fusion11',
            # '20190915_161631_conv_smoke_dust_relpos_rgb_fusion22',
            # '20190915_163216_conv_smoke_dust_relpos_rgb_fusion33',
            # '20190915_163540_conv_smoke_dust_relpos_rgb_fusion44',
            # Trained concatenated
            '20190922_165330_fusion0',
            '20190922_174343_fusion1',
            '20190922_175515_fusion2',
            '20190922_185620_fusion3',
            '20190922_210815_fusion0',
            '20190922_215725_fusion1',
            '20190922_222936_fusion2',
            '20190922_230722_fusion3',
            '20190923_013809_fusion4',
            '20190923_133805_fusion0',
        ]
        test_data = [args.set + '_smoke_dust_int_relpos_echo_fusion_close_cropped',args.set + '_smoke_dust_relpos_rgb_fusion_close']
    elif args.sensor == 'fusion_mid':
        args.name = [
            # '20190915_163555_conv_smoke_dust_relpos_rgb_fusion_mid00',
            # '20190915_174220_conv_smoke_dust_relpos_rgb_fusion_mid11',
            # '20190915_175143_conv_smoke_dust_relpos_rgb_fusion_mid22',
            # '20190915_180850_conv_smoke_dust_relpos_rgb_fusion_mid33',
            # '20190915_182847_conv_smoke_dust_relpos_rgb_fusion_mid44',
            # '20190915_195033_fusion_mid00',
            # '20190915_195910_fusion_mid11',
            # '20190915_200728_fusion_mid00',
            # '20190915_200732_fusion_mid11',
            # '20190915_201545_fusion_mid22',
            # '20190915_201755_fusion_mid33',
            # '20190915_203214_fusion_mid44',
            # trained concatenated
            '20190923_134014_fusion_mid0',
            '20190922_203613_fusion_mid2',
            '20190922_203625_fusion_mid1',
            '20190922_204424_fusion_mid3',
            '20190922_205219_fusion_mid4',
            '20190923_025042_fusion_mid0',
            '20190923_031843_fusion_mid1',
            '20190923_032059_fusion_mid2',
            '20190923_032316_fusion_mid3',
            '20190923_032741_fusion_mid4',
        ]
        test_data = [args.set + '_smoke_dust_int_relpos_echo_fusion_close_cropped',args.set + '_smoke_dust_relpos_rgb_fusion_close']

    args.name = args.name[:args.nb_en]
    if args.eval_title is None:
        args.eval_title = args.set + '_'+str(args.nb_en)+'en'
        if args.close:
            args.eval_title += '_close_40m'
        if 'fusion' in args.sensor:
            args.eval_title += ('_' + args.sensor)

    evaluate_model(args.sensor, args.name, set=args.set, eval_title=args.eval_title, test_data=test_data, save_pcl=args.save_pcl, save_voxel=args.save_voxel, voxel_eval=args.voxel_eval, missing_data=args.missing_data)

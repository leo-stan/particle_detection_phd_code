#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename: evaluate_lidar_stereo_fusion.py
Author: Leo Stanislas
Date Created: 06 Sep. 2019
Description: Evaluate the fusion between lidar and stereo camera predicted voxel maps
"""

import sys

sys.path.append('../..')

import os, os.path as osp
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, brier_score_loss, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
import tqdm
from particle_detection.src.config import cfg
from datetime import datetime
import argparse
import csv
from scipy.stats import entropy
from particle_detection.utils.asl_score import average_score_loss

def evaluate_lidar_stereo_fusion(lidar_pred, stereo_pred, eval_title, voxel_eval=True, entropy_threshold=None):
    if isinstance(lidar_pred, list) or isinstance(stereo_pred, list):
        assert isinstance(lidar_pred, list) == isinstance(stereo_pred, list)
        ensemble = False
    else:
        ensemble = True

    now = datetime.now()

    eval_dir = osp.join(cfg.LOG_DIR, 'fusion_logs/evals', now.strftime('%Y%m%d_%H%M%S') + '_' + eval_title)
    os.makedirs(eval_dir)

    # Load paths
    if ensemble:
        lidar_path = osp.join(cfg.LOG_DIR, 'dl_logs/ensemble_evaluations', lidar_pred)
        stereo_path = osp.join(cfg.LOG_DIR, 'st_logs/ensemble_evaluations', stereo_pred)
    else:
        lidar_path = osp.join(cfg.LOG_DIR, 'dl_logs', lidar_pred[0], 'evaluations', lidar_pred[1])
        stereo_path = osp.join(cfg.LOG_DIR, 'st_logs', stereo_pred[0], 'evaluations', stereo_pred[1])

    if voxel_eval:
        # Load voxel maps
        lidar_data = np.load(osp.join(lidar_path, 'scans_voxel.npy'))
        stereo_data = np.load(osp.join(stereo_path, 'scans_voxel.npy'))
    else:
        # Load point clouds
        lidar_data = np.load(osp.join(lidar_path, 'scans.npy'))
        stereo_data = np.load(osp.join(stereo_path, 'scans.npy'))
    assert len(lidar_data) == len(stereo_data)

    fused_scans = []
    fused_scans_proba = []
    fused_scans_bayes = []
    # Original sensor predictions for fused voxels
    lidar_scans = []
    stereo_scans = []

    # Voxels which haven't been matched and therefore fused
    lidar_scans_other = []
    stereo_scans_other = []

    for i, (ld, sd) in tqdm.tqdm(enumerate(zip(lidar_data, stereo_data)), total=len(lidar_data)):

        lidar_c = 0
        stereo_c = 0
        fused_map = np.array([]).reshape(0, lidar_data[0].shape[1] - 1)
        fused_map_proba = np.array([]).reshape(0, lidar_data[0].shape[1] - 1)
        fused_map_bayes = np.array([]).reshape(0, lidar_data[0].shape[1] - 1)
        lidar_map = np.array([]).reshape(0, lidar_data[0].shape[1] - 1)
        lidar_other = np.array([]).reshape(0, lidar_data[0].shape[1] - 1)
        stereo_map = np.array([]).reshape(0, lidar_data[0].shape[1] - 1)
        stereo_other = np.array([]).reshape(0, lidar_data[0].shape[1] - 1)
        while lidar_c < ld.shape[0] and stereo_c < sd.shape[0]:
            # Check X
            if ld[lidar_c, 0] == sd[stereo_c, 0]:
                # Check Y
                if ld[lidar_c, 1] == sd[stereo_c, 1]:
                    # Check Z
                    if ld[lidar_c, 2] == sd[stereo_c, 2]:
                        # Check Label
                        if ld[lidar_c, 3] == sd[stereo_c, 3]:

                            if entropy_threshold is not None:
                                # If both sensor predictions entropies are lower than theshold then skip
                                if entropy([1-ld[lidar_c, 5], ld[lidar_c, 5]]) < entropy_threshold and entropy([1-sd[stereo_c, 5], sd[stereo_c, 5]]) < entropy_threshold:
                                    lidar_c += 1
                                    stereo_c += 1
                                    continue

                            # Normal Fusion
                            fused_pred =(ld[lidar_c, 5] + sd[stereo_c, 5])/2
                            fused_pred = np.append(ld[lidar_c, :4], [(fused_pred > 0.5).astype(int), fused_pred])
                            fused_map = np.concatenate((fused_map, fused_pred.reshape(1, -1)), axis=0)

                            # Proba fusion
                            fused_pred_proba = (ld[lidar_c, 5] * sd[stereo_c, 6] + ld[lidar_c, 6] * sd[stereo_c, 5]) / (
                                    ld[lidar_c, 6] + sd[stereo_c, 6])
                            fused_pred_proba = np.append(ld[lidar_c, :4], [(fused_pred_proba > 0.5).astype(int), fused_pred_proba])
                            fused_map_proba = np.concatenate((fused_map_proba, fused_pred_proba.reshape(1, -1)), axis=0)

                            # Bayes fusion

                            # odds_l = np.log(ld[lidar_c, 5]/(1-ld[lidar_c, 5]))
                            # odds_s = np.log(sd[stereo_c, 5]/(1-sd[stereo_c, 5]))
                            #
                            # fused_pred_bayes = np.exp(odds_l+odds_s)
                            #
                            # fused_pred_bayes = np.append(ld[lidar_c, :4],
                            #                              [(fused_pred_bayes > 0.5).astype(int), fused_pred_bayes])
                            # fused_map_bayes = np.concatenate((fused_map_bayes, fused_pred_bayes.reshape(1, -1)), axis=0)

                            # Individual sensors
                            lidar_map = np.concatenate((lidar_map, ld[lidar_c, :6].reshape(1, -1)), axis=0)
                            stereo_map = np.concatenate((stereo_map, sd[stereo_c, :6].reshape(1, -1)), axis=0)
                        lidar_c += 1
                        stereo_c += 1
                    elif ld[lidar_c, 2] > sd[stereo_c, 2]:
                        stereo_other = np.concatenate((stereo_other, sd[stereo_c, :6].reshape(1, -1)), axis=0)
                        stereo_c += 1
                    else:
                        lidar_other = np.concatenate((lidar_other, ld[lidar_c, :6].reshape(1, -1)), axis=0)
                        lidar_c += 1
                elif ld[lidar_c, 1] > sd[stereo_c, 1]:
                    stereo_other = np.concatenate((stereo_other, sd[stereo_c, :6].reshape(1, -1)), axis=0)
                    stereo_c += 1
                else:
                    lidar_other = np.concatenate((lidar_other, ld[lidar_c, :6].reshape(1, -1)), axis=0)
                    lidar_c += 1
            elif ld[lidar_c, 0] > sd[stereo_c, 0]:
                stereo_other = np.concatenate((stereo_other, sd[stereo_c, :6].reshape(1, -1)), axis=0)
                stereo_c += 1
            else:
                lidar_other = np.concatenate((lidar_other, ld[lidar_c, :6].reshape(1, -1)), axis=0)
                lidar_c += 1

        fused_scans.append(fused_map)
        fused_scans_proba.append(fused_map_proba)
        lidar_scans.append(lidar_map)
        stereo_scans.append(stereo_map)
        lidar_scans_other.append(lidar_other)
        stereo_scans_other.append(stereo_other)
    np.save(osp.join(eval_dir, 'fused_voxel.npy'), fused_scans)
    np.save(osp.join(eval_dir, 'fused_voxel_proba.npy'), fused_scans_proba)
    np.save(osp.join(eval_dir, 'lidar_voxel.npy'), lidar_scans)
    np.save(osp.join(eval_dir, 'stereo_voxel.npy'), stereo_scans)
    np.save(osp.join(eval_dir, 'lidar_voxel_other.npy'), lidar_scans_other)
    np.save(osp.join(eval_dir, 'stereo_voxel_other.npy'), stereo_scans_other)
    fused_scans = np.concatenate(fused_scans)
    fused_scans_proba = np.concatenate(fused_scans_proba)
    lidar_scans = np.concatenate(lidar_scans)
    stereo_scans = np.concatenate(stereo_scans)

    with open(osp.join(eval_dir, 'eval_results.txt'), 'w') as f:

        print('Evaluation parameters:')
        print('Lidar pred: %s' % lidar_pred)
        print('Stereo pred: %s' % stereo_pred)
        print('nb_scans: %s' % len(lidar_data))
        f.write('Evaluation parameters:\n')
        f.write('Lidar pred: %s\n' % lidar_pred)
        f.write('Stereo pred: %s\n' % stereo_pred)
        f.write('nb_scans: %s\n' % len(lidar_data))

        for i in range(5):

            if i == 0:
                y_target = fused_scans[:, 3]
                y_pred = fused_scans[:, 4]
                y_mean_vec = fused_scans[:, 5]
                print('\n\nFusion Evaluation results:\n')
                f.write('\n\nFusion Evaluation results:\n')
                csv_name = 'normal_fusion.csv'
            elif i == 1:
                # Skip proba fusion if only one model in ensemble (no variance available = same are average fusion)
                # continue
                y_target = fused_scans_proba[:, 3]
                y_pred = fused_scans_proba[:, 4]
                y_mean_vec = fused_scans_proba[:, 5]
                print('\n\nProba Fusion Evaluation results:\n')
                f.write('\n\nProba Fusion Evaluation results:\n')
                csv_name = 'proba_fusion.csv'
            elif i == 2:
                continue
                # y_target = fused_scans_bayes[:, 3]
                # y_pred = fused_scans_bayes[:, 4]
                # y_mean_vec = fused_scans_bayes[:, 5]
                # print('\n\nBayes Fusion Evaluation results:\n')
                # f.write('\n\nBayes Fusion Evaluation results:\n')
                # csv_name = 'bayes_fusion.csv'
            elif i == 3:
                y_target = lidar_scans[:, 3]
                y_pred = lidar_scans[:, 4]
                y_mean_vec = lidar_scans[:, 5]
                print('\n\nLidar Evaluation results:\n')
                f.write('\n\nLidar Evaluation results:\n')
                csv_name = 'lidar.csv'
            else:
                y_target = stereo_scans[:, 3]
                y_pred = stereo_scans[:, 4]
                y_mean_vec = stereo_scans[:, 5]
                print('\n\nStereo Evaluation results:\n')
                f.write('\n\nStereo Evaluation results:\n')
                csv_name = 'stereo.csv'

            # Compute performance scores

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

            # Average score loss
            asl = average_score_loss(y_target, y_mean_vec)
            print('Average Score Loss: %f' % bsl)
            f.write('Average Score Loss: %f\n' % bsl)

            with open(osp.join(eval_dir, csv_name), 'a') as results:
                result_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                result_writer.writerow([precision_score(y_target,y_pred), recall_score(y_target,y_pred), f1_score(y_target,y_pred), bsl, asl])

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--lidar_name', type=str, help='log name')
    parser.add_argument('--stereo_name', type=str, help='log name')
    parser.add_argument('--eval_title', type=str, help='eval title')
    parser.add_argument('--voxel_eval', type=str, default='True', help='voxel eval flag')
    parser.add_argument('--entropy', type=float, help='entropy treshold')

    args = parser.parse_args()
    args.voxel_eval = args.voxel_eval == 'True'

    # args.eval_title = 'test'
    # args.lidar_name = 'eval_20190906_122133_visualisation'
    # args.stereo_name = 'eval_20190906_122532_visualisation'

    evaluate_lidar_stereo_fusion(args.lidar_name, args.stereo_name, args.eval_title, args.voxel_eval, entropy_threshold=args.entropy)

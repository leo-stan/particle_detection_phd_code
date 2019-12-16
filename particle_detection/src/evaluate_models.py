#!/usr/bin/env python

import sys

sys.path.insert(0, '../../')

import os
import os.path as osp
import numpy as np
import torch
import tqdm
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, brier_score_loss, precision_recall_curve
import yaml
import particle_detection.models as models
from particle_dataset import ParticleDataset
from particle_detection.utils.segment_scan import segment_scan
from config import cfg
from datetime import datetime
from sklearn.externals import joblib
from torch.utils.data import DataLoader
from torch.autograd import Variable
import math
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


def evaluate_model(sensor, input_model, eval_title=None, set='testing', test_data=None, save_pcl=False, save_voxel=False, proba_pred=True,
                   pr_curve=True, dos=0, voxel_eval=True):
    if sensor == 'lidar':
        datasets_dir = osp.join(cfg.DATASETS_DIR, 'dl_datasets')
        logs_dir = osp.join(cfg.LOG_DIR, 'dl_logs')
    else:
        datasets_dir = osp.join(cfg.DATASETS_DIR, 'st_datasets')
        logs_dir = osp.join(cfg.LOG_DIR, 'st_logs')

    # Load dataset parameters
    with open(osp.join(logs_dir, input_model, 'config.yaml'), 'r') as f:
        log_params = yaml.load(f, Loader=yaml.SafeLoader)

    # Evaluation parameters
    if not eval_title:
        eval_title = set
    if voxel_eval:
        eval_title += '_voxel'
    if not test_data:
        test_data = log_params['train_data'].replace('training', set)

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
    # os.makedirs(osp.join(eval_dir,'scans'))

    criterion = nn.CrossEntropyLoss()
    # Choose device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    if parameters['conv']:
        if parameters['dos']:
            model = models.SmokeNet3DConvDOS(model_config=parameters["model_config"],
                                             map_config=ds_parameters["map_config"])
        else:
            model = models.SmokeNet3DConv(model_config=parameters["model_config"],
                                          map_config=ds_parameters["map_config"],batch_size=1)
    else:
        model = models.SmokeNetNoConv(model_config=parameters["model_config"], map_config=ds_parameters["map_config"])
    # Load the best model saved
    model_state = osp.join(logs_dir, input_model, 'best.pth.tar')
    model_state = torch.load(model_state)
    model.load_state_dict(model_state['model_state_dict'])
    model.eval()
    model.to(device)

    # ParticleDataset
    scaler = joblib.load(osp.join(logs_dir, input_model, 'scaler.pkl'))

    y_target = []
    y_pred = []
    y_mean_vec = []
    y_std_vec = []
    val_loss = 0

    eval_pcl = []
    eval_voxel = []

    test_loader = DataLoader(ParticleDataset(dataset_dir=osp.join(datasets_dir, test_data), scaler=scaler),
                             batch_size=1, shuffle=False, num_workers=4, collate_fn=detection_collate)

    sm = nn.Softmax(dim=1)
    pred_time = 0
    for s, sample in tqdm.tqdm(enumerate(test_loader), total=len(test_loader),
                               desc='Evaluating scan', ncols=80,
                               leave=False):

        inputs = Variable(torch.FloatTensor(sample['inputs']))
        labels = Variable(torch.LongTensor(sample['labels']))

        inputs, labels = inputs.to(device), labels.to(device)

        if parameters["conv"]:
            coords = Variable(torch.LongTensor(sample['coords']))
            coords = coords.to(device)
            if parameters["dos"] and dos > 0:
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
                with torch.no_grad():
                    # start = time.time()
                    pred = sm(model(inputs, coords))
                    # end = time.time()
                    # pred_time += end - start
        else:
            with torch.no_grad():
                pred = sm(model(inputs))

        if parameters["dos"] and dos > 0:
            proba_mean = pred.mean(dim=0)
            proba_std = pred.std(dim=0)[:, 1].cpu().data.numpy()
            pred = proba_mean.argmax(dim=1).cpu().data.numpy()
            proba_mean = proba_mean[:, 1].cpu().data.numpy()
            if not voxel_eval or save_pcl:
                pcl = np.load(osp.join(datasets_dir, test_data, 'scan_pcls', str(s) + '.npy'))
                if proba_pred:
                    raw_pred_points = segment_scan(pcl, pred, ds_parameters['map_config'], proba_mean, proba_std)
                else:
                    raw_pred_points = segment_scan(pcl, pred, ds_parameters['map_config'])
            # np.save(osp.join(eval_dir, 'scans', str(s)), raw_pred_points[:, :7])
        else:

            [_, index] = pred.max(dim=1)

            if proba_pred:
                proba = pred[:, 1].cpu().data.numpy()

            pred = index.cpu().data.numpy()

            # convert predicted voxels back into pointcloud
            if not voxel_eval or save_pcl:
                pcl = np.load(osp.join(datasets_dir, test_data, 'scan_pcls', str(s) + '.npy'))
                if proba_pred:
                    raw_pred_points = segment_scan(pcl, pred, ds_parameters['map_config'], output_proba_mean=proba)
                else:
                    raw_pred_points = segment_scan(pcl, pred, ds_parameters['map_config'])

        # Removes any scan with ground truth = 1 for whole scan
        if np.sum((sample['labels'] == 1).astype(np.int8)) / float(sample['labels'].shape[0]) < 0.5:
            if save_voxel:
                if proba_pred:
                    eval_voxel.append(np.concatenate((sample['coords'][:, 1:] * np.array((ds_parameters['map_config'][
                                                                                              'voxel_size_x'],
                                                                                          ds_parameters['map_config'][
                                                                                              'voxel_size_y'],
                                                                                          ds_parameters['map_config'][
                                                                                              'voxel_size_z'])),
                                                      sample['labels'].reshape(-1, 1),
                                                      pred.reshape(-1, 1), proba.reshape(-1, 1)), axis=1))
                else:
                    eval_voxel.append(
                        np.concatenate((sample['coords'][:, 1:], sample['labels'].reshape(-1, 1), pred.reshape(-1, 1)),
                                       axis=1))
            if save_pcl:
                if parameters["dos"] and dos > 0:
                    eval_pcl.append(raw_pred_points)
                else:
                    if sensor == 'lidar':
                        eval_pcl.append(raw_pred_points[:, :8])
                    else:
                        eval_pcl.append(raw_pred_points[:, :7])
            if voxel_eval:
                if proba_pred:
                    y_mean_vec.append(proba)
                y_pred.append(pred)
                y_target.append(sample['labels'])
            else:
                if proba_pred:
                    if sensor == 'lidar':
                        y_mean_vec.append(raw_pred_points[:, 7])
                        if parameters["dos"] and dos > 0:
                            y_std_vec.append(raw_pred_points[:, 8])
                    else:
                        y_mean_vec.append(raw_pred_points[:, 6])
                        if parameters["dos"] and dos > 0:
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
        if parameters["dos"] and dos > 0:
            y_std_vec = np.concatenate(y_std_vec)
    y_target = np.concatenate(y_target).astype(np.uint8)

    y_mean_vec = y_mean_vec[y_target != 2]
    y_pred = y_pred[y_target != 2]
    y_target = y_target[y_target != 2]

    pred_time /= float(ds_parameters['nb_scans'])
    with open(osp.join(eval_dir, 'eval_results.txt'), 'w') as f:

        f.write('Evaluation parameters:\n')
        f.write('ParticleDataset: %s\n' % test_data)
        f.write('nb_scans: %s\n' % ds_parameters['nb_scans'])
        f.write('avg_pred_time: %f\n' % pred_time)
        f.write('dataset_size: %s\n' % ds_parameters['dataset_size'])
        f.write('Voxel Eval: %r\n' % voxel_eval)
        f.write('Dropout Sampling: %d\n' % dos)
        f.write('\n\nEvaluation results:\n')

        # Compute performance scores
        print('\n')
        print("Evaluation results for model: %s" % input_model)

        print('average pred time %s' % str(pred_time))

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

        # Negative Log-Likelihood
        nll = 0
        for i in range(y_target.shape[0]):
            if y_target[i].astype(int) == 1:
                nll += -math.log(max(0.001, y_mean_vec[i]))
            else:
                nll += -math.log(max(0.001, 1 - y_mean_vec[i]))
        nll /= y_target.shape[0]
        print('Negative Log Likelihood: %f' % nll)
        f.write('Negative Log Likelihood: %f\n' % nll)

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


if __name__ == '__main__':
    # Models
    # input_models = [
    # '20190614_170037_both_int_relpos_echo',
    # '20190616_152017_smoke_dust_int_relpos_echo_no_conv',
    # '20190616_155112_smoke_dust_int_relpos_echo_no_conv'
    # '20190406_161014_3d_conv',
    # '20190614_160717_both_int_relpos_echo',
    # '20190614_170037_both_int_relpos_echo',
    # '20190614_170133_both_int_relpos_echo',
    # '20190614_170909_both_int_relpos_echo',
    # '20190615_113731_both_int_relpos_echo',
    # '20190619_192645_smoke_dust_int_relpos_echo_conv',
    # '20190619_195345_smoke_dust_int_relpos_echo_conv'
    # '20190620_170540_smoke_dust_int_relpos_echo_conv_dos_p05_noBN1D',
    # '20190620_142815_smoke_dust_int_relpos_echo_conv_dos_p05_noBN1D',
    # '20190620_171539_smoke_dust_int_relpos_echo_conv_dos_p05_noBN1D'
    # '20190621_114451_smoke_dust_int_relpos_echo_conv_dos_p05_noBN1D'
    # '20190621_174226_smoke_dust_int_relpos_echo_conv_dos_p05_withBN'
    # '20190624_160803_smoke_dust_int_relpos_echo_conv_dos_p05_withBN',
    # '20190624_161359_smoke_dust_int_relpos_echo_conv_dos_p05_withBN',
    # '20190624_173228_smoke_dust_int_relpos_echo_conv_dos_p05_withBN'
    # '20190620_141301_smoke_dust_int_relpos_echo_conv_dos_p0',
    # '20190629_162620_smoke_trained',
    # '20190629_162757_dust_trained',
    # '20190629_162852_dust_trained',
    # '20190701_140215_smoke_dust_int_relpos_echo',
    # '20190701_140416_smoke_dust_int_relpos_echo',
    # '20190701_140555_smoke_dust_int_relpos_echo',
    # '20190701_140742_smoke_trained',
    # '20190701_140922_smoke_trained',
    # '20190701_141406_smoke_trained',
    # '20190701_141949_dust_trained',
    # '20190701_142809_dust_trained',
    # '20190701_174413_smoke_trained',
    # '20190701_174415_smoke_trained'
    # '20190702_174625_no_forest',
    # '20190702_174935_no_forest',
    # '20190702_175240_no_forest',
    # '20190702_180220_no_forest',
    # '20190702_180920_no_forest',
    # '20190704_171618_test_no_softmax',
    # '20190704_172047_test_no_softmax'
    # '20190706_131200_smoke_trained_FCdos',
    # '20190706_131339_smoke_trained_FCdos'
    # '20190706_140318_smoke_dust_int_relpos_2'
    # '20190706_141301_smoke_dust_int_relpos_2'
    # '20190706_183919_dust_trained',
    # '20190706_184018_dust_trained',
    # '20190706_185127_dust_trained',
    # '20190706_185135_dust_trained',
    # '20190706_192056_dust_trained',
    # '20190706_195536_dust_trained',
    # '20190706_200418_dust_trained'
    # '20190719_170106_conv_smoke_dust_int',
    # '20190719_170711_conv_smoke_dust_int_relpos',
    # '20190719_171309_conv_smoke_dust_int_relpos_echo',
    # '20190719_182417_conv_smoke_dust_echo',
    # '20190719_182919_conv_smoke_dust_int_echo',
    # '20190719_194757_conv_smoke_dust_relpos',
    # '20190719_195148_conv_smoke_dust_relpos_echo',
    # ]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--name', type=str, help='log name')
    parser.add_argument('--set', type=str, help='e.g. "testing"')
    parser.add_argument('--sensor', type=str, help='sensor')
    parser.add_argument('--test_data', type=str, help='test data')
    parser.add_argument('--eval_title', type=str, help='eval title')
    parser.add_argument('--save_pcl', action='store_true', help='save predicted pcl flag')
    parser.add_argument('--save_voxel', action='store_true', help='save predicted voxel flag')
    parser.add_argument('--voxel_eval', type=str, help='voxel eval flag')

    args = parser.parse_args()

    if args.set == 'visualisation':
        args.save_pcl = True
        args.save_voxel = True
    args.voxel_eval = args.voxel_eval == 'True'
    # sensor = 'lidar'
    # sensor = 'stereo'

    # Evaluation for prediction time
    # eval_title = '12-smoke'
    # test_data = 'visualisation_smoke_dust_relpos_rgb'
    # args.name = '20190809_121706_conv_smoke_dust_relpos_rgb_real1'
    # args.sensor = 'stereo'

    # test_data = 'visualisation_smoke_dust_relpos_rgb'
    # args.name = '20190720_222224_conv_smoke_dust_int_relpos_echo'
    # args.sensor = 'lidar'
    # test_data = 'testing_smoke_int_relpos_echo'
    # eval_title = 'smoke_eval'
    # 'testing_smoke_int_relpos_echo'

    # LIDAR VOXEL SIZE PREDICTION TIME EVAL
    # args.eval_title = 'pred_time'
    # args.voxel_eval = True
    # args.sensor = 'lidar'
    # # 30cm
    # args.test_data = 'visualisation_smoke_dust_int_relpos_echo_30cm'
    # args.name = '20190824_113911_conv_smoke_dust_int_relpos_echo_30cm0'
    # evaluate_model(args.sensor, args.name, set=args.set, eval_title=args.eval_title, test_data=args.test_data,
    #                save_pcl=args.save_pcl, save_voxel=args.save_voxel, voxel_eval=args.voxel_eval)
    # # 25cm
    # args.test_data = 'visualisation_smoke_dust_int_relpos_echo_25cm'
    # args.name = '20190824_190445_conv_smoke_dust_int_relpos_echo_25cm1'
    # evaluate_model(args.sensor, args.name, set=args.set, eval_title=args.eval_title, test_data=args.test_data,
    #                save_pcl=args.save_pcl, save_voxel=args.save_voxel, voxel_eval=args.voxel_eval)
    # # 20cm
    # args.test_data = 'visualisation_smoke_dust_int_relpos_echo'
    # args.name = '20190720_222224_conv_smoke_dust_int_relpos_echo'
    # evaluate_model(args.sensor, args.name, set=args.set, eval_title=args.eval_title, test_data=args.test_data,
    #                save_pcl=args.save_pcl, save_voxel=args.save_voxel, voxel_eval=args.voxel_eval)
    # # 15cm
    # args.test_data = 'visualisation_smoke_dust_int_relpos_echo_15cm'
    # args.name = '20190826_125752_conv_smoke_dust_int_relpos_echo_15cm0'
    # evaluate_model(args.sensor, args.name, set=args.set, eval_title=args.eval_title, test_data=args.test_data,
    #                save_pcl=args.save_pcl, save_voxel=args.save_voxel, voxel_eval=args.voxel_eval)
    # # 10cm
    # args.test_data = 'visualisation_smoke_dust_int_relpos_echo_10cm'
    # args.name = '20190826_182335_conv_smoke_dust_int_relpos_echo_10cm0'
    # evaluate_model(args.sensor, args.name, set=args.set, eval_title=args.eval_title, test_data=args.test_data,
    #                save_pcl=args.save_pcl, save_voxel=args.save_voxel, voxel_eval=args.voxel_eval)

    # LIDAR VOXEL SIZE PREDICTION TIME EVAL
    # args.eval_title = 'pred_time'
    # args.voxel_eval = True
    # args.sensor = 'stereo'
    # # 30cm
    # args.test_data = 'visualisation_smoke_dust_relpos_rgb_30cm'
    # args.name = '20190824_153701_conv_smoke_dust_relpos_rgb_30cm0'
    # evaluate_model(args.sensor, args.name, set=args.set, eval_title=args.eval_title, test_data=args.test_data,
    #                save_pcl=args.save_pcl, save_voxel=args.save_voxel, voxel_eval=args.voxel_eval)
    # # 25cm
    # args.test_data = 'visualisation_smoke_dust_relpos_rgb_25cm'
    # args.name = '20190824_234903_conv_smoke_dust_relpos_rgb_25cm0'
    # evaluate_model(args.sensor, args.name, set=args.set, eval_title=args.eval_title, test_data=args.test_data,
    #                save_pcl=args.save_pcl, save_voxel=args.save_voxel, voxel_eval=args.voxel_eval)
    # # 20cm
    # args.test_data = 'visualisation_smoke_dust_relpos_rgb'
    # args.name = '20190809_121706_conv_smoke_dust_relpos_rgb_real1'
    # evaluate_model(args.sensor, args.name, set=args.set, eval_title=args.eval_title, test_data=args.test_data,
    #                save_pcl=args.save_pcl, save_voxel=args.save_voxel, voxel_eval=args.voxel_eval)
    # # 15cm
    # args.test_data = 'visualisation_smoke_dust_relpos_rgb_15cm'
    # args.name = '20190825_095922_conv_smoke_dust_relpos_rgb_15cm0'
    # evaluate_model(args.sensor, args.name, set=args.set, eval_title=args.eval_title, test_data=args.test_data,
    #                save_pcl=args.save_pcl, save_voxel=args.save_voxel, voxel_eval=args.voxel_eval)
    # # 10cm
    # args.test_data = 'visualisation_smoke_dust_relpos_rgb_10cm'
    # args.name = '20190826_115507_conv_smoke_dust_relpos_rgb_10cm0'
    # evaluate_model(args.sensor, args.name, set=args.set, eval_title=args.eval_title, test_data=args.test_data,
    #                save_pcl=args.save_pcl, save_voxel=args.save_voxel, voxel_eval=args.voxel_eval)

    # LIDAR 10cm eval

    # args.sensor = 'lidar'
    # args.set = 'visualisation'
    # args.eval_title = '6_dust_start'
    # args.test_data = '6_dust_start'
    # args.name = '20190720_222224_conv_smoke_dust_int_relpos_echo'
    # # args.pcl_eval = True
    # args.save_voxel = True
    # args.save_pcl = True
    # args.voxel_eval = True
    # names = [
    #     '20190826_182335_conv_smoke_dust_int_relpos_echo_10cm0',
    #     '20190826_182622_conv_smoke_dust_int_relpos_echo_10cm1',
    #     '20190826_182622_conv_smoke_dust_int_relpos_echo_10cm2',
    #     '20190826_184037_conv_smoke_dust_int_relpos_echo_10cm3',
    #     '20190827_004726_conv_smoke_dust_int_relpos_echo_10cm4',
    #     ]
    # for name in names:
    #     args.name = name
    evaluate_model(args.sensor, args.name, set=args.set, eval_title=args.eval_title, test_data=args.test_data, save_pcl=args.save_pcl, save_voxel=args.save_voxel,voxel_eval=args.voxel_eval)

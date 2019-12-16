#!/usr/bin/env python



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys

sys.path.insert(0, '../../')
import particle_detection.models as models
from particle_dataset import ParticleDataset, ParticleFusionDataset
from config import cfg
import os
from datetime import datetime
import os.path as osp
from smokenet_trainer import SmokeNetTrainer
import yaml
from sklearn.externals import joblib
import numpy as np
import argparse


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


def train_model(sensor, train_data, val_data, conv, output_model, seed=None):
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

    # Dropout sampling
    dos = False

    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
    else:
        seed = 'None'

    now = datetime.now()
    criterion = nn.CrossEntropyLoss()

    if 'fusion' in sensor:
        with open(osp.join(datasets_dir[0], train_data[0], 'config.yaml'), 'r') as f:
            dataset_params = yaml.load(f, Loader=yaml.SafeLoader)
        with open(osp.join(datasets_dir[1], train_data[1], 'config.yaml'), 'r') as f:
            stereo_dataset_params = yaml.load(f, Loader=yaml.SafeLoader)

        features = dataset_params['features'] + stereo_dataset_params['features']
        assert dataset_params['map_config'] == stereo_dataset_params['map_config'], 'map config mismatch'
        map_config = dataset_params['map_config']
        features_size = dataset_params['features_size'] + stereo_dataset_params['features_size']

    else:
        with open(osp.join(datasets_dir, train_data, 'config.yaml'), 'r') as f:
            dataset_params = yaml.load(f, Loader=yaml.SafeLoader)

            features = dataset_params['features']
            map_config = dataset_params['map_config']
            features_size = dataset_params['features_size']

    # Check what hardware is available
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    print('Device used: ' + device.type)

    # Parameters
    parameters = {
        "resume_model": '',
        "conv": conv,
        "dos": dos,
        "train_data": train_data,
        "val_data": val_data,
        "max_iterations": 10000,
        "interval_validate": 500,
        "batch_size": 4,
        "features": features,
        "map_config": map_config,
        "model_config": {
            'features_size': features_size,
            'VFE1_OUT': 32,
            'VFE2_OUT': 128
        },
        "optimiser": {
            "name": 'Adam',
            "lr": 0.001,
            "betas": (0.9, 0.999),
            "eps": 1e-08,
            "weight_decay": 0,
        },
        "random_seed": seed,
        "fusion_match_only": False
    }

    output_path = osp.join(logs_dir, now.strftime('%Y%m%d_%H%M%S') + '_' + output_model)

    # Save training parameters
    os.makedirs(output_path)

    if 'fusion' in sensor:
        train_data[0] = osp.join(datasets_dir[0], train_data[0])
        train_data[1] = osp.join(datasets_dir[1], train_data[1])
        val_data[0] = osp.join(datasets_dir[0], val_data[0])
        val_data[1] = osp.join(datasets_dir[1], val_data[1])

        # Datasets
        train_loader = DataLoader(ParticleFusionDataset(lidar_dataset_dir=train_data[0],stereo_dataset_dir=train_data[1], use_dataset_scaler=True, match_only=parameters['fusion_match_only']),
                                  batch_size=parameters['batch_size'], shuffle=False, num_workers=4,
                                  collate_fn=detection_collate)

        val_loader = DataLoader(ParticleFusionDataset(lidar_dataset_dir=val_data[0],stereo_dataset_dir=val_data[1], lidar_scaler=train_loader.dataset.lidar_scaler,stereo_scaler=train_loader.dataset.stereo_scaler, match_only=parameters['fusion_match_only']),
                                batch_size=parameters['batch_size'], shuffle=False, num_workers=4,
                                collate_fn=detection_collate)

    else:
        train_data = osp.join(datasets_dir, train_data)
        val_data = osp.join(datasets_dir, val_data)

        # Datasets
        train_loader = DataLoader(ParticleDataset(dataset_dir=train_data, use_dataset_scaler=True),
                                  batch_size=parameters['batch_size'], shuffle=False, num_workers=4,
                                  collate_fn=detection_collate)

        val_loader = DataLoader(ParticleDataset(dataset_dir=val_data, scaler=train_loader.dataset.scaler),
                                batch_size=parameters['batch_size'], shuffle=False, num_workers=4,
                                collate_fn=detection_collate)

    # Models
    if conv:
        if dos:
            model = models.SmokeNet3DConvDOS(model_config=parameters['model_config'],
                                             map_config=parameters['map_config']).to(device)
        else:
            if sensor == 'fusion_mid':
                model = models.SmokeNet3DConvMid(model_config=parameters['model_config'],
                                              map_config=parameters['map_config'],
                                              batch_size=parameters['batch_size']).to(device)
            else:
                model = models.SmokeNet3DConv(model_config=parameters['model_config'],
                                          map_config=parameters['map_config'], batch_size=parameters['batch_size']).to(device)
    else:
        model = models.SmokeNetNoConv(model_config=parameters['model_config'], map_config=parameters['map_config']).to(
            device)

    if parameters['resume_model'] != '':
        checkpoint = torch.load(osp.join(logs_dir, parameters['resume_model'], 'checkpoint.pth.tar'))
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        start_epoch = 0
        start_iteration = 0

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=parameters['optimiser']['lr'], betas=parameters['optimiser']['betas'],
                           eps=parameters['optimiser']['eps'], weight_decay=parameters['optimiser']['weight_decay'],
                           amsgrad=False)
    if parameters['resume_model'] != '':
        optimizer.load_state_dict(checkpoint['optim_state_dict'])

    # Save model parameters
    with open(osp.join(output_path, 'config.yaml'), 'w') as f:
        yaml.safe_dump(parameters, f, default_flow_style=False)
    if 'fusion' in sensor:
        joblib.dump(train_loader.dataset.lidar_scaler, osp.join(output_path, 'lidar_scaler.pkl'))
        joblib.dump(train_loader.dataset.stereo_scaler, osp.join(output_path, 'stereo_scaler.pkl'))
    else:
        joblib.dump(train_loader.dataset.scaler, osp.join(output_path, 'scaler.pkl'))

    # Start training
    trainer = SmokeNetTrainer(
        device=device,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        out=output_path,
        max_iter=parameters['max_iterations'],
        interval_validate=parameters['interval_validate'],
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--feature_id', type=int, default=0, help='features id')
    parser.add_argument('--unique_id', type=int, default=0, help='unique id')
    parser.add_argument('--sensor', type=str, help='sensor')
    parser.add_argument('--data_suffix', type=str, default='', help='data_suffix')

    args = parser.parse_args()

    # particles = [
    #     # 'smoke',
    #     # 'dust',
    #     'smoke_dust'
    # ]
    particles = 'smoke_dust'

    # args.data_suffix = '_10cm'
    sensor = args.sensor
    # sensor = 'lidar'
    # sensor = 'stereo'

    if sensor == 'lidar':
        features = [
            ['int', 'relpos', 'echo'],
            ['int'],
            ['echo'],
            ['relpos'],
            ['int', 'relpos'],
            ['int', 'echo'],
            ['relpos', 'echo'],
        ]
    else:
        features = [
            ['relpos', 'rgb'],
            ['relpos'],
            ['rgb'],
        ]
    conv = True

    dataset_name = ''
    for f_dataset_name in features[args.feature_id]:
        dataset_name += '_' + f_dataset_name
    if conv:
        name = 'conv_' + particles + dataset_name
    else:
        name = 'noconv_' + particles + dataset_name

    if 'fusion' in sensor:
        # suffix must be 0 1 2 3 4
        test_data = ['training_smoke_dust_int_relpos_echo_fusion_sync' + str(args.unique_id), 'training_smoke_dust_relpos_rgb_fusion_sync' + str(args.unique_id)]
        val_data = ['validation_smoke_dust_int_relpos_echo_fusion', 'validation_smoke_dust_relpos_rgb_fusion_sync']
        name = sensor
    else:
        test_data = 'training_' + particles + dataset_name + args.data_suffix
        val_data = 'validation_' + particles + dataset_name + args.data_suffix
    train_model(sensor, test_data, val_data, conv, name + args.data_suffix + str(args.unique_id), seed=args.unique_id)

    # Single model debug
    # train_model('lidar', 'training_smoke_dust_int_relpos_echo', 'validation_smoke_dust_int_relpos_echo', conv, 'test')

    # Fusion debug
    # train_model('fusion_mid', ['11_smoke_early_fusion_test','11_smoke_early_fusion_test'], ['11_smoke_early_fusion_test','11_smoke_early_fusion_test'], conv, 'test')

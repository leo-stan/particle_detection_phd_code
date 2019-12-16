#!/usr/bin/env python

import os.path as osp

import numpy as np
import yaml
from sklearn.externals import joblib
from torch.utils.data import Dataset


def match_data(lidar_data, lidar_coords, lidar_labels, stereo_data, stereo_coords, stereo_labels):
    lidar_c = 0
    stereo_c = 0
    lidar_map_id = []
    stereo_map_id = []
    while lidar_c < lidar_coords.shape[0] and stereo_c < stereo_coords.shape[0]:
        # Check X
        if lidar_coords[lidar_c, 0] == stereo_coords[stereo_c, 0]:
            # Check Y
            if lidar_coords[lidar_c, 1] == stereo_coords[stereo_c, 1]:
                # Check Z
                if lidar_coords[lidar_c, 2] == stereo_coords[stereo_c, 2]:
                    # Check Label
                    if lidar_labels[lidar_c] == stereo_labels[stereo_c]:
                        lidar_map_id.append(lidar_c)
                        stereo_map_id.append(stereo_c)
                    lidar_c += 1
                    stereo_c += 1
                elif lidar_coords[lidar_c, 2] > stereo_coords[stereo_c, 2]:
                    stereo_c += 1
                else:
                    lidar_c += 1
            elif lidar_coords[lidar_c, 1] > stereo_coords[stereo_c, 1]:
                stereo_c += 1
            else:
                lidar_c += 1
        elif lidar_coords[lidar_c, 0] > stereo_coords[stereo_c, 0]:
            stereo_c += 1
        else:
            lidar_c += 1
    return np.concatenate((lidar_data[lidar_map_id], stereo_data[stereo_map_id]), axis=2), lidar_coords[lidar_map_id], lidar_labels[lidar_map_id]


def concatenate_data(lidar_data, lidar_coords, lidar_labels, stereo_data, stereo_coords, stereo_labels):
    lidar_c = 0
    stereo_c = 0

    lidar_shape = lidar_data[0].shape
    stereo_shape = stereo_data[0].shape

    conc_data = []
    conc_coords = []
    conc_labels = []

    while lidar_c < lidar_coords.shape[0] and stereo_c < stereo_coords.shape[0]:
        # Check X
        if lidar_coords[lidar_c, 0] == stereo_coords[stereo_c, 0]:
            # Check Y
            if lidar_coords[lidar_c, 1] == stereo_coords[stereo_c, 1]:
                # Check Z
                if lidar_coords[lidar_c, 2] == stereo_coords[stereo_c, 2]:
                    # Check Label
                    if lidar_labels[lidar_c] == stereo_labels[stereo_c]:

                        conc_data.append(np.concatenate((lidar_data[lidar_c], stereo_data[stereo_c]), axis=1))
                        conc_labels.append(lidar_labels[lidar_c])
                        conc_coords.append(lidar_coords[lidar_c])

                    lidar_c += 1
                    stereo_c += 1
                elif lidar_coords[lidar_c, 2] > stereo_coords[stereo_c, 2]:
                    conc_data.append(np.concatenate((np.zeros(lidar_shape),stereo_data[stereo_c]), axis=1))
                    conc_labels.append(stereo_labels[stereo_c])
                    conc_coords.append(stereo_coords[stereo_c])
                    stereo_c += 1
                else:
                    conc_data.append(np.concatenate((lidar_data[lidar_c],np.zeros(stereo_shape)), axis=1))
                    conc_labels.append(lidar_labels[lidar_c])
                    conc_coords.append(lidar_coords[lidar_c])
                    lidar_c += 1
            elif lidar_coords[lidar_c, 1] > stereo_coords[stereo_c, 1]:
                conc_data.append(np.concatenate((np.zeros(lidar_shape), stereo_data[stereo_c]), axis=1))
                conc_labels.append(stereo_labels[stereo_c])
                conc_coords.append(stereo_coords[stereo_c])
                stereo_c += 1
            else:
                conc_data.append(np.concatenate((lidar_data[lidar_c], np.zeros(stereo_shape)), axis=1))
                conc_labels.append(lidar_labels[lidar_c])
                conc_coords.append(lidar_coords[lidar_c])
                lidar_c += 1
        elif lidar_coords[lidar_c, 0] > stereo_coords[stereo_c, 0]:
            conc_data.append(np.concatenate((np.zeros(lidar_shape), stereo_data[stereo_c]), axis=1))
            conc_labels.append(stereo_labels[stereo_c])
            conc_coords.append(stereo_coords[stereo_c])
            stereo_c += 1
        else:
            conc_data.append(np.concatenate((lidar_data[lidar_c], np.zeros(stereo_shape)), axis=1))
            conc_labels.append(lidar_labels[lidar_c])
            conc_coords.append(lidar_coords[lidar_c])
            lidar_c += 1

    return np.asarray(conc_data), np.asarray(conc_coords), np.asarray(conc_labels)

class ParticleDataset(Dataset):
    """Lidar dataset"""

    def __init__(self, dataset_dir, use_dataset_scaler = False, scaler = None):

        self.data = []
        self.labels = []
        self.dataset_dir = dataset_dir

        with open(osp.join(dataset_dir, 'config.yaml'), 'r') as f:
            self.params = yaml.load(f,Loader=yaml.SafeLoader)
        # If scaler provided then use it (prediction), otherwise use dataset scaler (training)
        self.scaler = None
        if scaler:
            self.scaler = scaler
        elif use_dataset_scaler:
            self.scaler = joblib.load(osp.join(dataset_dir, 'scaler.pkl'))
        self.features = self.params['features']

    def __len__(self):
        return self.params['nb_scans']

    def __getitem__(self, idx):

        data = np.load(osp.join(self.dataset_dir, 'scan_voxels', 'voxels_' + str(idx) + '.npy'))
        if self.scaler:
            if 'sensor' in self.params: # For backward compatibility with earlier datasets
                if self.params['sensor'] == 'lidar_hc':
                    data = self.scaler.transform(data)
                else:
                    data = self.scaler.transform(data.reshape(-1, data.shape[2])).reshape(-1, data.shape[1],
                                                                                          data.shape[2])
            else:
                data = self.scaler.transform(data.reshape(-1,data.shape[2])).reshape(-1,data.shape[1],data.shape[2])
        labels = np.load(osp.join(self.dataset_dir, 'scan_voxels', 'labels_' + str(idx) + '.npy'))
        coords = np.load(osp.join(self.dataset_dir, 'scan_voxels', 'coords_' + str(idx) + '.npy'))

        sample = {'inputs': data,'coords': coords, 'labels': labels}

        return sample


class ParticleFusionDataset(Dataset):
    """Lidar dataset"""

    def __init__(self, lidar_dataset_dir, stereo_dataset_dir, use_dataset_scaler = False, lidar_scaler = None, stereo_scaler = None, match_only=True):

        self.data = []
        self.labels = []
        self.lidar_dataset_dir = lidar_dataset_dir
        self.stereo_dataset_dir = stereo_dataset_dir
        self.match_only = match_only

        with open(osp.join(lidar_dataset_dir, 'config.yaml'), 'r') as f:
            self.lidar_params = yaml.load(f,Loader=yaml.SafeLoader)

        # If scaler provided then use it (prediction), otherwise use dataset scaler (training)
        self.lidar_scaler = None
        if lidar_scaler:
            self.lidar_scaler = lidar_scaler
        elif use_dataset_scaler:
            self.lidar_scaler = joblib.load(osp.join(lidar_dataset_dir, 'scaler.pkl'))
        self.lidar_features = self.lidar_params['features']

        # Stereo sensor
        with open(osp.join(stereo_dataset_dir, 'config.yaml'), 'r') as f:
            self.stereo_params = yaml.load(f,Loader=yaml.SafeLoader)

        # If scaler provided then use it (prediction), otherwise use dataset scaler (training)
        self.stereo_scaler = None
        if stereo_scaler:
            self.stereo_scaler = stereo_scaler
        elif use_dataset_scaler:
            self.stereo_scaler = joblib.load(osp.join(stereo_dataset_dir, 'scaler.pkl'))
        self.stereo_features = self.stereo_params['features']

        # Check that datasets have the same characteristic
        assert self.lidar_params['nb_scans'] == self.stereo_params['nb_scans'], 'Number of scan mismatch'
        assert self.lidar_params['map_config'] == self.stereo_params['map_config'], 'Map config mismatch'
        assert self.lidar_params['datasets'] == self.stereo_params['datasets'], 'Datasets mismatch'

    def __len__(self):
        return self.lidar_params['nb_scans']

    def __getitem__(self, idx):

        lidar_data = np.load(osp.join(self.lidar_dataset_dir, 'scan_voxels', 'voxels_' + str(idx) + '.npy'))
        stereo_data = np.load(osp.join(self.stereo_dataset_dir, 'scan_voxels', 'voxels_' + str(idx) + '.npy'))

        if self.lidar_scaler:
            if self.lidar_params['sensor'] == 'lidar_hc':
                lidar_data = self.lidar_scaler.transform(lidar_data)
            else:
                lidar_data = self.lidar_scaler.transform(lidar_data.reshape(-1,lidar_data.shape[2])).reshape(-1,lidar_data.shape[1],lidar_data.shape[2])

        if self.stereo_scaler:
            if self.stereo_params['sensor'] == 'lidar_hc':
                stereo_data = self.stereo_scaler.transform(stereo_data)
            else:
                stereo_data = self.stereo_scaler.transform(stereo_data.reshape(-1,stereo_data.shape[2])).reshape(-1,stereo_data.shape[1],stereo_data.shape[2])

        lidar_labels = np.load(osp.join(self.lidar_dataset_dir, 'scan_voxels', 'labels_' + str(idx) + '.npy'))
        stereo_labels = np.load(osp.join(self.stereo_dataset_dir, 'scan_voxels', 'labels_' + str(idx) + '.npy'))
        lidar_coords = np.load(osp.join(self.lidar_dataset_dir, 'scan_voxels', 'coords_' + str(idx) + '.npy'))
        stereo_coords = np.load(osp.join(self.stereo_dataset_dir, 'scan_voxels', 'coords_' + str(idx) + '.npy'))

        # Only return voxels that match in position and label
        if self.match_only:
            fused_data, fused_coords, fused_labels = match_data(lidar_data, lidar_coords, lidar_labels, stereo_data, stereo_coords, stereo_labels)
        else:
            fused_data, fused_coords, fused_labels = concatenate_data(lidar_data, lidar_coords, lidar_labels,
                                                    stereo_data,
                                                    stereo_coords,
                                                    stereo_labels)

        return {'inputs': fused_data, 'coords': fused_coords, 'labels': fused_labels}




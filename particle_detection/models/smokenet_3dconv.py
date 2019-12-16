#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable


# Fully Connected Network
class FCN(nn.Module):

    def __init__(self,cin,cout):
        super(FCN, self).__init__()
        self.cout = cout
        self.linear = nn.Linear(cin, cout)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self,x):
        # KK is the stacked k across batch
        kk, t, _ = x.shape
        x = self.linear(x.view(kk*t,-1))
        x = func.relu(self.bn(x))
        return x.view(kk,t,-1)

# Voxel Feature Encoding layer
class VFE(nn.Module):

    def __init__(self,cin,cout,map_config):
        super(VFE, self).__init__()
        assert cout % 2 == 0
        self.units = cout // 2
        self.fcn = FCN(cin,self.units)
        self.map_config = map_config

    def forward(self, x, mask):
        # point-wise feature
        pwf = self.fcn(x)
        #locally aggregated feature
        laf = torch.max(pwf,1)[0].unsqueeze(1).repeat(1,self.map_config['voxel_pt_count'],1)
        # point-wise concat feature
        pwcf = torch.cat((pwf,laf),dim=2)
        # apply mask
        mask = mask.unsqueeze(2).repeat(1, 1, self.units * 2)
        pwcf = pwcf * mask.float()

        return pwcf

# conv3d + bn + relu
class Conv3d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, batch_norm=True):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)

        return func.relu(x, inplace=True)

# Convolutional Middle Layer
class CML(nn.Module):
    def __init__(self):
        super(CML, self).__init__()
        self.conv3d_1 = Conv3d(128, 64, 3, s=(1, 1, 1), p=(1, 1, 1))
        self.conv3d_2 = Conv3d(64, 64, 3, s=(1, 1, 1), p=(1, 1, 1))
        self.conv3d_3 = Conv3d(64, 2, 3, s=(1, 1, 1), p=(1, 1, 1))

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        return x

class SmokeNet3DConv(nn.Module):
    def __init__(self, model_config, map_config, batch_size):
        super(SmokeNet3DConv, self).__init__()

        self.bs = batch_size
        self.map_config = map_config
        self.VFE1 = VFE(model_config['features_size'],model_config['VFE1_OUT'],map_config)
        self.VFE2 = VFE(model_config['VFE1_OUT'], model_config['VFE2_OUT'],map_config)
        self.fcn = FCN(model_config['VFE2_OUT'],model_config['VFE2_OUT'])
        self.fcf = FCN(model_config['VFE2_OUT'],2)
        self.cml = CML()
        # self.sm = nn.Softmax(dim=1)

    def voxel_indexing(self, sparse_features, coords):
        dim = sparse_features.shape[-1]

        dense_feature = Variable(torch.zeros(dim, self.bs, int(self.map_config['map_size_x'] / self.map_config['voxel_size_x']), int(self.map_config['map_size_y'] / self.map_config['voxel_size_y']), int(self.map_config['map_size_z'] / self.map_config['voxel_size_z'])).cuda())

        dense_feature[:, coords[:,0], coords[:,1], coords[:,2], coords[:,3]]= sparse_features.transpose(0,1)

        return dense_feature.transpose(0, 1)

    def voxel_deindexing(self, dense_feature, coords):

        sparse_features = dense_feature[coords[:,0],:, coords[:,1], coords[:,2], coords[:,3]]

        return sparse_features


    def forward(self, voxel_features, voxel_coords):
        mask = torch.ne(torch.max(voxel_features, 2)[0], 0)
        x = self.VFE1(voxel_features, mask)
        x = self.VFE2(x, mask)
        x = self.fcn(x)
        # Voxel Feature Vector [1,128]
        x = torch.max(x, 1)[0]

        x = self.voxel_indexing(x,voxel_coords)

        # 3d convolutions
        x = self.cml(x)

        res = self.voxel_deindexing(x,voxel_coords)

        # res = self.sm(x)

        return res


class SmokeNet3DConvMid(nn.Module):
    def __init__(self, model_config, map_config, batch_size):
        super(SmokeNet3DConvMid, self).__init__()

        self.bs = batch_size
        self.map_config = map_config
        self.VFE1_lidar = VFE(7,model_config['VFE1_OUT'],map_config)
        self.VFE2_lidar = VFE(model_config['VFE1_OUT'], model_config['VFE2_OUT'],map_config)

        self.VFE1_stereo = VFE(6, model_config['VFE1_OUT'], map_config)
        self.VFE2_stereo = VFE(model_config['VFE1_OUT'], model_config['VFE2_OUT'], map_config)

        self.fcn = FCN(model_config['VFE2_OUT']*2,model_config['VFE2_OUT'])
        self.fcf = FCN(model_config['VFE2_OUT'], 2)
        self.cml = CML()
        # self.sm = nn.Softmax(dim=1)

    def voxel_indexing(self, sparse_features, coords):
        dim = sparse_features.shape[-1]

        dense_feature = Variable(torch.zeros(dim, self.bs, int(self.map_config['map_size_x'] / self.map_config['voxel_size_x']), int(self.map_config['map_size_y'] / self.map_config['voxel_size_y']), int(self.map_config['map_size_z'] / self.map_config['voxel_size_z'])).cuda())

        dense_feature[:, coords[:,0], coords[:,1], coords[:,2], coords[:,3]]= sparse_features.transpose(0,1)

        return dense_feature.transpose(0, 1)

    def voxel_deindexing(self, dense_feature, coords):

        sparse_features = dense_feature[coords[:,0],:, coords[:,1], coords[:,2], coords[:,3]]

        return sparse_features

    def forward(self, voxel_features, voxel_coords):

        x_lidar = voxel_features[:,:,:7]
        x_stereo = voxel_features[:,:,7:]

        mask_lidar = torch.ne(torch.max(x_lidar, 2)[0], 0)
        x_lidar = self.VFE1_lidar(x_lidar, mask_lidar)
        x_lidar = self.VFE2_lidar(x_lidar, mask_lidar)

        mask_stereo = torch.ne(torch.max(x_stereo, 2)[0], 0)
        x_stereo = self.VFE1_stereo(x_stereo, mask_stereo)
        x_stereo = self.VFE2_stereo(x_stereo, mask_stereo)
        x = self.fcn(torch.cat((x_lidar, x_stereo),dim=2))
        # Voxel Feature Vector [1,128]
        x = torch.max(x, 1)[0]

        x = self.voxel_indexing(x,voxel_coords)

        # 3d convolutions
        x = self.cml(x)

        res = self.voxel_deindexing(x,voxel_coords)

        # res = self.sm(x)

        return res
#!/usr/bin/env python

""""
    File name: smokenet_no_conv.py
    Author: Leo Stanislas
    Date created: 2018/08/14
    Python Version: 2.7
"""

import torch
import torch.nn as nn
import torch.nn.functional as func


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

class SmokeNetNoConv(nn.Module):
    def __init__(self, model_config, map_config):
        super(SmokeNetNoConv, self).__init__()

        self.VFE1 = VFE(model_config['features_size'], model_config['VFE1_OUT'], map_config)
        self.VFE2 = VFE(model_config['VFE1_OUT'], model_config['VFE2_OUT'], map_config)
        self.fcn = FCN(model_config['VFE2_OUT'], model_config['VFE2_OUT'])
        self.fcf = FCN(model_config['VFE2_OUT'], 2)
        self.sm = nn.Softmax(dim=2)

    def forward(self, x):
        mask = torch.ne(torch.max(x, 2)[0], 0)
        x = self.VFE1(x, mask)

        # VFE_geometry = self.VFE_geometry(x[:,:,:3],mask)
        # VFE_intensity = self.VFE_intensity(x[:,:,4:5],mask)
        # VFE_echo = self.VFE_echo(x[:,:,-3:],mask)
        # VFE1 = torch.cat((VFE_geometry,VFE_intensity,VFE_echo),2)

        x = self.VFE2(x, mask)
        # VFE3 = self.VFElayer3(VFE2, mask).view(-1, cfg.VOXEL_POINT_COUNT, cfg.VFE3_OUT)

        # version with maxpool at the end
        # VFV = self.fc_f(self.do_f(VFE2)).view(-1, cfg.VOXEL_POINT_COUNT, 2)
        # res = self.sm(torch.max(VFV, dim=1, keepdim=True)[0]).view(-1, 1, 2)

        # Voxel Feature Vector [1,128]
        x = torch.max(x, dim=1, keepdim=True)[0]
        res = self.sm(self.fcf(x))

        return res.view(-1, 2)

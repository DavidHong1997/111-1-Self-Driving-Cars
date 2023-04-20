import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl
from models.subnets.subnets import MultiheadAttention, MLP, MapNet, SubGraph
import yaml 

import math
import matplotlib.pyplot as plt

''' Yaml Parser
'''
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
_output_heads = int(config['net']['output_heads'])

''' Model
'''
class Baseline(pl.LightningModule):
    def __init__(self):
        super(Baseline, self).__init__()
        ''' history state (x, y, vx, vy, yaw, object_type) * 5s * 10Hz
        '''
        self.history_encoder = MLP(250, 128, 128)

        self.lane_encoder = MapNet(2, 128, 128, 10)
        self.lane_attn = MultiheadAttention(128, 8)

        self.neighbor_encoder = MapNet(6,128,128, 11)
        self.neighbor_attn = MultiheadAttention(128, 8)
        
        trajs = []
        confs = []
        ''' we predict 6 different future trajectories to handle different possible cases.
        '''
        for i in range(6):
            ''' future state (x, y, vx, vy, yaw) * 6s * 10Hz
            '''
            trajs.append(
                MLP(128, 256, 300)
                )
            ''' we use model to predict the confidence score of prediction
            '''
            confs.append(
                    nn.Sequential(
                    MLP(128, 64, 1),
                    nn.Sigmoid()
                    )
                )
        self.future_decoder_traj = nn.ModuleList(trajs)
        self.future_decoder_conf = nn.ModuleList(confs)

    def forward(self, data):
        ''' In deep learning, data['x'] means input, data['y'] means groundtruth
        '''
        #x = data['x'].reshape(-1, ...)
        px = data['x'][:,:,0]
        py = data['x'][:,:,1]

        for i in range(0,48):
            vx = (px[:,i+1] - px[:,i]) / 0.1
            vy = (py[:,i+1] - py[:,i]) / 0.1
            data['x'][:,i+1,2] = vx
            data['x'][:,i+1,3] = vy

        for j in range(0,48):
            data['x'][:,j+1,0] = data['x'][:,j,0] + data['x'][:,j,2] * 0.1
            data['x'][:,j+1,1] = data['x'][:,j,1] + data['x'][:,j,3] * 0.1
 
        x = data['x'][:,:,:5]
        x = x.reshape(-1, 250)
        
        x = self.history_encoder(x)
        	
        lane = data['lane_graph']
        lane = self.lane_encoder(lane)

        neighbor = data['neighbor_graph']
        neighbor = self.neighbor_encoder(neighbor)
        
        x = x.unsqueeze(0)
        lane = lane.unsqueeze(0)
        neighbor = neighbor.unsqueeze(0)

        lane_mask = data['lane_mask']
        neighbor_mask = data['neighbor_mask']
        lane_attn_out = self.lane_attn(x, lane, lane, attn_mask=lane_mask) 
        neighbor_attn_out = self.neighbor_attn(x, neighbor, neighbor, attn_mask=neighbor_mask)
	
        x = x + lane_attn_out + neighbor_attn_out
        x = x.squeeze(0)
        
        trajs = []
        confs = []
        for i in range(6):
            trajs.append(self.future_decoder_traj[i](x))
            confs.append(self.future_decoder_conf[i](x))
        trajs = torch.stack(trajs, 1)
        confs = torch.stack(confs, 1)
        
        return trajs, confs
	

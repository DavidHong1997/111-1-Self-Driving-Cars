import yaml 

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl
from av2.datasets.motion_forecasting.eval.metrics import compute_ade, compute_fde, compute_brier_fde
import ipdb

#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

''' Yaml Parser
'''
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


''' Configure constants
'''
CONF_ALPHA = int(config['criterion']['conf_alpha'])
TRAJ_BETA = int(config['criterion']['traj_beta'])
OUTPUT_HEADS = int(config['net']['output_heads'])
OUT_DIM = int(config['net']['out_dim'])
_OBS_STEPS = int(config['constant']['obs_steps'])
_PRED_STEPS = int(config['constant']['pred_steps'])
_TOTAL_STEPS = int(config['constant']['total_steps'])

# wrapper of all goal loss
class ArgoCriterion:
    def __init__(self):
        pass

    def reshape(self, pred, cls, y):
        pred = pred.reshape(-1, 60, 5).detach().cpu().numpy()
        pred = pred[..., :2]
        cls = cls.reshape(-1).detach().cpu().numpy()
        y = y.reshape(60, 5)
        y = y[..., :2].detach().cpu().numpy()
        return pred, cls, y

    def forward(self, pred, cls, y): 
        pred, cls, y = self.reshape(pred, cls, y)
        
        #Input: (K, N, 2), (N, 2) -> Output: (K,)
        ade_k = compute_ade(pred, y)
        #Input: (K, N, 2), (N, 2) -> Output: (K,)
        fde_k = compute_fde(pred, y)
        #Input: (K, N, 2), (N, 2), (K,) -> Output: (K,)
        b_fde_k = compute_brier_fde(pred, y, cls, normalize=False)
                                                                                
        idx = np.argmax(cls) 
        ade_1 = ade_k[idx]
        fde_1 = fde_k[idx]
        ade_6_min = ade_k.min()
        fde_6_min = fde_k.min()
        b_fde_6_min = b_fde_k.min()

        loss_dict = {
            'ade_1': ade_1,
            'fde_1': fde_1,
            'ade_6_min': ade_6_min,
            'fde_6_min': fde_6_min,
            'b_fde_6_min': fde_6_min,
        }
        return loss_dict

class Criterion(pl.LightningModule):
    def __init__(self):
        super(Criterion, self).__init__()
        self.output_heads = OUTPUT_HEADS 

    def forward(self, pred, cls, y, object_type, valid=None):
        ''' Optimize of diverse trajectory
        '''
        fde, arg_min = minFDE(
                pred, 
                y, 
                object_type,
                self.output_heads, 
                valid
                )
        ade = minADE(
                pred, 
                y, 
                object_type,
                self.output_heads,
                valid
                )
        
        '''
        Optimize confidence of the minFDE trajectory 
        '''
        maxmargin = MaxMarginLoss(
                arg_min, 
                cls, 
                y,
                object_type,
                valid
                )
        
        '''
        Total Loss Structure
        '''
        loss = ade + fde + CONF_ALPHA * maxmargin
        loss_dict = {
            'total_loss': loss,
            'min_ade': ade,
            'min_fde': fde,
            'max_margin': maxmargin,
        }
        return loss_dict

def object_type_output(loss, object_type):
    object_type_loss = []
    loss = loss.float()
    for i in range(0, 10):
        test_single_loss = loss[object_type==i]
        if test_single_loss.shape[0] == 0:
            torch.tensor(0).to(device)
        else:
            single_loss = test_single_loss.mean() 
        object_type_loss.append(single_loss)
    object_type_loss = torch.stack(object_type_loss)
    '''
    "vehicle": 0,
    "pedestrian": 1,
    "motorcyclist": 2,
    "cyclist": 3,
    "bus": 4,
    "static": 5,
    "background": 6,
    "construction": 7,
    "riderless_bicycle": 8,
    "unknown": 9,
    '''
    object_weight = torch.Tensor([1, 1, 1, 1, 1, 0.2, 0.2, 0.2, 0.2, 0.0]).to(device)
    object_type_loss = object_type_loss @ object_weight / object_weight.sum()
    return object_type_loss

def MaxMarginLoss(
        arg_min, 
        cls, y, 
        object_type, 
        valid=None
    ):
    cls = cls.squeeze(-1)
    min_cfd = cls.gather(1, arg_min.unsqueeze(1))
    loss = cls-min_cfd+0.2
    loss = F.relu(loss)
    mask = torch.ones_like(loss).scatter_(1, arg_min.unsqueeze(1),0)
    loss = loss[mask.bool()]
    return loss.mean()

def minADE(
        out, 
        y, 
        object_type, 
        output_heads=6, 
        valid=None, 
        key_frame=-1
    ):
    ''' Reshape
    '''
    # sample['y'].shape -> 60 * [x,y,vx,vy] = (60, 5)
    y = y.reshape(-1, 1, _PRED_STEPS, OUT_DIM)
    if key_frame == -1:
        # (batch, output_heads, 240) 
        out = out.reshape(-1, output_heads, _PRED_STEPS, OUT_DIM)
    else:
        y = y[:,:,(_PRED_STEPS//key_frame)-1::(_PRED_STEPS//key_frame),:]
        out = out.reshape(-1, output_heads, key_frame, OUT_DIM)
    valid = valid if valid is not None else torch.ones(y.shape[:-1]).to(device)
    valid = valid.repeat(1, output_heads, 1)

    ''' Calculate Error
    '''
    # calculate square error
    loss = (y - out) ** 2

    # sum x and y [batch, output_heads, time, coord] => [batch, output_heads, time]
    loss = loss.sum(-1)

    # square root square error to get displacement error
    loss = torch.sqrt(loss+1e-10)

    # 1e-10 prevent division by zero error
    loss = loss.sum(-1) / (valid.sum(-1)+1e-10)

    # find minimum loss prediction, [batch, pred] => [batch]
    loss, arg_min = loss.min(-1)
    return loss.mean()

def minFDE(
        out, 
        y, 
        object_type, 
        output_heads=6, 
        valid=None
    ):
    '''
    Reshape
    '''
    # sample['y'].shape -> 60 * [x,y,vx,vy] = (60, 5)
    y = y.reshape(-1, 1, _PRED_STEPS, OUT_DIM)
    # (batch, output_heads, 240) 
    out=out.reshape(-1, output_heads, _PRED_STEPS, OUT_DIM)
    valid = valid if valid is not None else torch.ones(y.shape[:-1]).to(device)
    valid = valid.repeat(1, output_heads, 1); 

    ''' Calculate only distance
    '''
    #y = y[..., :2]
    #out = out[..., :2]
    
    '''
    Calculate Error
    '''
    # calculate square error
    loss = (y - out) ** 2

    # sum x and y [batch, pred, time, coord] => [batch, pred, time]
    loss = loss.sum(-1)

    # square root square error to get displacement error
    loss = torch.sqrt(loss)

    # calculate last time stamp but only valid => [batch, pred]
    loss = loss[..., -1]

    # find minimum loss prediction, [batch, pred] => [batch]
    loss, arg_min = loss.min(-1)

    return loss.mean(), arg_min

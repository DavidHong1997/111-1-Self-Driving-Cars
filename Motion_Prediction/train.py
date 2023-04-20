'''
Boosted by lightning module
'''
import os 
import yaml 
import argparse
from pathlib import Path

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dataset import Argoverse2Dataset, argo_multi_agent_collate_fn
from dataset import KungFuDataset, kf_multi_agent_collate_fn
from models.baseline import Baseline
from criterion import ArgoCriterion, Criterion
from utils import VisualizeInterface

''' Argparse
'''
parser = argparse.ArgumentParser()
parser.add_argument(
        "--train",
        help="train mode",
        action='store_true',
        )
parser.add_argument(
        "--val",
        help="val mode",
        action='store_true',
        )
parser.add_argument(
        "--dev",
        help="fast dev mode",
        action='store_true',
        )
parser.add_argument(
        "--viz",
        help="viz mode",
        action='store_true',
        )
parser.add_argument(
        "--argo",
        help="argo mode",
        action='store_true',
        )
parser.add_argument(
        "--kungfu",
        help="kungfu mode",
        action='store_true',
        )
parser.add_argument(
        "--ckpt",
        help="ckpt name",
        default="",
        type=str
        )
args = parser.parse_args()

''' Yaml Parser
'''
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

''' Configure constants
'''
WARMUP_EPOCH = int(config['optimizer']['warmup_epoch'])
LR = float(config['optimizer']['lr'])
WD = float(config['optimizer']['wd'])
GAMMA = float(config['scheduler']['gamma'])

SAVE_TOP_K = int(config['trainer']['save_top_k'])
MAX_EPOCHS = int(config['trainer']['max_epochs'])
BATCH_SIZE = int(config['trainer']['batch_size'])

root = config['argo_data']['root']
val_dir = Path(root) / Path('raw/validation')

processed_val_dir = Path(root) / Path('processed/validation/')
processed_val_dir.mkdir(parents=True, exist_ok=True)

ckpt_dir = Path(config['repo']['ckpt_dir'])
ckpt_path = ckpt_dir / Path(str(args.ckpt))

kf_root = Path(config['kf_data']['root'])
kf_tracks = Path(config['kf_data']['tracking'])
kf_map = Path(config['kf_data']['map'])

kf_processed = Path(kf_root) / Path('processed/')
kf_processed.mkdir(parents=True, exist_ok=True)

class MotionPrediction(pl.LightningModule):
    def __init__(self):
        super().__init__()

        ''' Data
        '''
        self.dataset = Argoverse2Dataset
        self.collate_fn = argo_multi_agent_collate_fn
        self.kf_dataset = KungFuDataset
        self.kf_collate_fn = kf_multi_agent_collate_fn

        ''' Model
        '''
        self.model = Baseline()
        
        ''' Metric
        '''
        self.criterion = Criterion()
        self.argo_criterion = ArgoCriterion()
        
        ''' Init loss
        '''
        self.loss = 0
        self.g_ade = 0
        self.g_cons = 0
        self.ade = 0
        self.fde = 0
        self.max_margin = 0
        self.b_fde_6_min = 0 
        self.steps = 0
        
        ''' Visualization
        '''
        self.visualize_interface = VisualizeInterface()
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LR, weight_decay=WD)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=GAMMA)
        return [self.optimizer], [self.lr_scheduler]

    def training_step(self, train_batch, train_idx):
        ''' inference
        '''
        out, cls = self.model(train_batch)
        
        ''' loss
        '''
        y = train_batch['y']
        object_type = train_batch['object_type']
        loss_dict = self.criterion(out, cls, y, object_type)
        
        ''' logging
        '''
        self.loss += loss_dict['total_loss'].item()
        self.ade += loss_dict['min_ade'].item()
        self.fde += loss_dict['min_fde'].item()
        self.max_margin += loss_dict['max_margin'].item()
        self.steps += 1
        self.log("lr", (self.optimizer).param_groups[0]['lr'])
        self.log("train_loss", self.loss/self.steps, prog_bar=True)
        self.log("ade", self.ade/self.steps, prog_bar=True)
        self.log("fde", self.fde/self.steps, prog_bar=True)
        self.log("max_margin", self.max_margin/self.steps, prog_bar=True)
        
        return loss_dict['total_loss']

    def training_epoch_end(self, outputs):
        ''' clear running loss after one epoch end
        '''
        self.loss = 0
        self.g_ade = 0
        self.g_cons = 0
        self.ade = 0
        self.fde = 0
        self.max_margin = 0
        self.b_fde_6_min = 0 
        self.steps = 0
    
    def test_step(self, test_batch, test_idx):
        ''' inference
        '''
        out, cls = self.model(test_batch)
        
        
        ''' groundtruth
        '''
        y = test_batch['y']
        object_type = test_batch['object_type']
        
        ''' visualize
        '''
        if args.argo:
            self.visualize_interface.argo_forward(
                test_idx, self.train_dataset,
                test_batch, self.model, 
                out, cls
            )
        elif args.kungfu:
            self.visualize_interface.kf_forward(
                test_idx, test_batch, 
                out, cls)

    def prepare_data(self):
        self.train_dataset = self.dataset(val_dir, processed_val_dir)
        self.test_kf_dataset = self.kf_dataset(kf_tracks, kf_map, kf_processed)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=self.collate_fn,
            num_workers=8,
            pin_memory=True,
        ) 
    
    def test_dataloader(self):
        if args.argo:
            return DataLoader(
                self.train_dataset,
                batch_size=1,
                collate_fn=self.collate_fn,
                num_workers=0,
            )
        elif args.kungfu:
            return DataLoader(
                self.test_kf_dataset,
                batch_size=1,
                collate_fn=self.kf_collate_fn,
                num_workers=0,
            ) 

if __name__ == '__main__':
    if args.train:
        wandb_logger = WandbLogger(project="motion-prediction", offline=True)
        if args.ckpt != "": 
            model = MotionPrediction.load_from_checkpoint(
                    checkpoint_path=ckpt_path,
                    map_location=None,
            )
        else:
            model = MotionPrediction()
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir, 
            filename='{epoch:02d}-{train_loss:.2f}',
            save_top_k=SAVE_TOP_K, 
            mode="min",
            monitor="train_loss"
            )
        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            accelerator="cpu",
            max_epochs=MAX_EPOCHS,
            logger=wandb_logger,
            fast_dev_run = True if args.dev else False
        )
        trainer.fit(model)
    elif args.viz:
        if args.ckpt:
            model = MotionPrediction.load_from_checkpoint(
                    checkpoint_path=ckpt_path,
                    map_location=None,
            )
        else:
            model = MotionPrediction()
        trainer = pl.Trainer(
                accelerator="cpu"
                )
        trainer.test(model)





from pickletools import optimize
import numpy as np
import torch
import wandb
import random
from pdb import set_trace as st
from tqdm import tqdm
import time

from .misc import *
from .meters import *

class Trainer:
    def __init__(self, args, start_epoch, train_dataloader, val_dataloader, model, criterion, criterion_2, optimizer, scheduler, saver, device, epoch_size, logger, **kwargs):
        self.access_list=[]
        self.metric_list=[]
        self.args = args
        self.epoch_size = epoch_size
        self.model = model

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.saver = saver

        self.criterion = criterion
        self.criterion_2 = criterion_2
        self.device = device
        self.logger = logger
        self.start_epoch = start_epoch

    def _init_record(self):
        results=[]
        for i,j in zip(self.access_list, self.metric_list):
            exec(f"self.{i} = AverageMeter('{j}')")
            results.append(eval(f"self.{i}"))
        self.progress = ProgressMeter(self.epoch_size, results)
    
    def _init_record_val(self):
        results=[]
        for i,j in zip(self.val_access_list, self.val_metric_list):
            exec(f"self.{i} = AverageMeter('{j}')")
            results.append(eval(f"self.{i}"))
        self.progress = ProgressMeter(self.epoch_size, results)
    
    def _train_forward(self,datas_pos,datas_neg):
        raise NotImplementedError

    def _val_forward(self,datas):
        raise NotImplementedError

    def _update_metric(self):
        raise NotImplementedError
    
    def _train_metric(self):
        raise NotImplementedError

    def _val_metric(self):
        raise NotImplementedError
    
    def _init_val_metric(self):
        raise NotImplementedError
    
    def _model_forward(self):
        raise NotImplementedError


    def _train(self, epoch):
        self._init_record()
        self.model.train()

        for i, datas in enumerate(self.train_dataloader):
            # datas_real = datas["real"].numpy()
            # np.save(f"{self.args.exam_dir}/datas_real_{epoch}_{i}.npy", datas_real)
            # datas_fake = datas["fake"].numpy()
            # np.save(f"{self.args.exam_dir}/datas_fake_{epoch}_{i}.npy", datas_fake)
            
            self.optimizer.zero_grad()
            self._train_forward(datas)
            self.optimizer.step()
            self.scheduler.step()
            
            self._update_metric()

            self.total_steps += 1

            if self.args.local_rank == 0:
                if i % self.args.train.print_interval == 0:
                    self.logger.info(f'Epoch-{epoch}: {self.progress.display(i)}')

            if self.total_steps >= 2000 and self.total_steps % 2000 == 0:
                self._validate(self.total_steps)
                self.model.train()

        self._train_metric(epoch)
    
    def _validate(self,epoch):
        self._init_record_val()
        self.model.eval()
        self._init_val_metric()
        for i, datas in enumerate(tqdm(self.val_dataloader)):
            with torch.no_grad():
                self._val_forward(datas)
        
        self._val_metric(epoch)

    def running(self):
        self.total_steps = 0
        for epoch in range(self.start_epoch, self.args.train.epochs + 1):
            if self.args.distributed:
                self.train_dataloader.sampler.set_epoch(epoch)
            
            self._train(epoch)

        if self.args.local_rank == 0:
            wandb.finish()
            self.logger.info(self.args)


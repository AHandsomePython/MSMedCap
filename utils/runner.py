import logging
import os
import time
import yaml
import torch


class RunnerBase:
    def __init__(
        self,
        model,
        optimizer,
        dataloader,
        max_epoch,
        device
    ):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train_step(self, samples):
        output = self.model(samples)
        return output['loss']

    def train_epoch(self):
        for samples in self.dataloader:
            with torch.cuda.amp.autocast(enabled=True):
                loss = self.train_step(samples)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
    
    def train(self):
        start_time = time.time()
        logging.info("Start training")
        self.training_start_hook()
        for cur_epoch in range(self.max_epoch):
            # trainning phase
            info = {
                'cur_epoch': cur_epoch
            }
            self.epoch_start_hook(info)
            logging.info("Training on epoch {}".format(cur_epoch))
            self.train_epoch()
            self.epoch_end_hook(info)
    
    def training_start_hook(self):
        pass

    def epoch_start_hook(self, info):
        pass

    def epoch_end_hook(self, info):
        pass



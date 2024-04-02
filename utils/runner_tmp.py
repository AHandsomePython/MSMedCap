import logging
import os
import time
import yaml
import torch

from utils.data import build_dataloader

class RunnerBase:
    def __init__(self, model, cfg):
        self.config = self.build_config(cfg)
        self.max_epoch = self.config["run"]["max_epoch"]
        self.device = self.config["run"]["device"]
        
        self.model = model.to(self.device)
        self.dataloader = build_dataloader(self.config)
        self.optimizer = self.build_optimizer()
        self.scaler = torch.cuda.amp.GradScaler()
        return
    
    def build_config(self, cfg):
        with open(cfg, 'r') as file:
            _config = yaml.load(file, Loader=yaml.FullLoader)
        return _config
    
    def build_optimizer(self):
        lr_scale = self.config["run"]["lr_layer_decay"]
        weight_decay = self.config["run"]["weight_decay"]
        optim_params = self.model.get_optimizer_params(weight_decay, lr_scale)
        # optim_params = self.model.Parameters()
        

        num_parameters = 0
        for p_group in optim_params:
            for p in p_group["params"]:
                num_parameters += p.data.nelement()    
        logging.info("number of trainable parameters: {}".format(num_parameters))      
                
        beta2 = self.config["run"]["beta2"]

        _optimizer = torch.optim.AdamW(
            optim_params,
            lr=float(self.config["run"]["init_lr"]),
            betas=(0.9, beta2),
        )    
        return _optimizer
    
    def train(self):
        start_time = time.time()
        
        logging.info("Start training")
        for cur_epoch in range(self.max_epoch):
            # trainning phase
            logging.info("Training on epoch {}".format(cur_epoch))
            self.train_epoch()
            
        return
    
    def train_epoch(self):
        for samples in self.dataloader:
            with torch.cuda.amp.autocast(enabled=True):
                loss = self.train_step(samples)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        return
    
    def train_step(self,samples):
        output = self.model(samples)
        return # loss
    
    
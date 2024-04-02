# Dataset and Dataloader

import torch
from torch.utils.data import Dataset, DataLoader
import os
import csv
import numpy as np
import yaml
import json
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import sys
sys.path.append("..")


from utils.runner import *
from tqdm import tqdm
from models.mlp import ClassficationMLP

# dataset_base = "/data2/xcg_data/lavis_data/2023us/features"
jsonpath = "/home/xcg/medical-research/Project23us/labels/8000patient.json"
dataset_base = "/data2/xcg_data/lavis_data/2023us/features"
# dataset_base = "/data2/xcg_data/lavis_data/Breast_images/features"
# jsonpath = "/home/xcg/medical-research/Project23us/labels/breast_patient.json"
# csvpath = "/data/xcg/lavis_data/coco-2023us/excels/translated.csv" # ignored
# Define your custom dataset class
class Dataset_2023us(Dataset):
    def __init__(self):
        self.jsonpath = jsonpath
        self.dataset_base = dataset_base
        self.limitation = 8
        with open(jsonpath , 'r') as f:
            self.data = json.load(f)
        # self.keylist = list(self.data.keys())
        self.searchset = self.load_searchset()
        self.pairs, self.keylist = self.load_pairs()
    
    def load_searchset(self):
        searchset = {}
        imageids = os.listdir( self.dataset_base + "/clip_features")
        for imageid in imageids:
            personid = imageid.split("_")[1]
            if personid not in searchset.keys():
                searchset[personid] = [imageid]
            else:
                searchset[personid].append(imageid)
        return searchset
    
    def load_pairs(self):
        # {personid: [[image_ids, ], [probabilities...  3*11 ] ], }
        pairs = {}
        keylist = []
        for personid in self.data.keys():
            # if personid in self.searchset.keys():
            #     imglist = self.searchset[personid]
            #     probabilitylist = []
            #     for organ in self.data[personid].keys():
            #         for mark in self.data[personid][organ].keys():
            #             probabilitylist+=self.data[personid][organ][mark]                    
            
            #     pairs[personid] = [imglist, probabilitylist]
            #     keylist.append(personid)
            try:
            # print(len(self.data[personid].keys()))
                imglist = self.searchset[personid]
                probabilitylist = []
                for organ in self.data[personid].keys():
                    
                    probabilitylist+=self.data[personid][organ]["good"]
                pairs[personid] = [imglist, probabilitylist]
                keylist.append(personid)

            except:
                continue
        return pairs, keylist

    def __len__(self):
        return len(self.keylist)

    def __getitem__(self, index):
        # clip_feature, sam_feature, caption
        personid = self.keylist[index]
        clip_feature = None
        sam_feature = None
        pairlen = len(self.pairs[personid][0])
        if pairlen <= self.limitation:
            for i in range(self.limitation):
                clip_feature_path = self.dataset_base + "/clip_features/" + self.pairs[personid][0][i % pairlen] 
                sam_feature_path = self.dataset_base + "/sam_features/" + self.pairs[personid][0][i % pairlen] 
                clip_dataloads = np.load(clip_feature_path)
                sam_dataloads = np.load(sam_feature_path)
                if clip_feature is None:
                    clip_feature = torch.from_numpy(clip_dataloads["arr"]).unsqueeze(0)
                    sam_feature = torch.from_numpy(sam_dataloads["arr"]).unsqueeze(0)
                else:
                    clip_feature = torch.cat([clip_feature, torch.from_numpy(clip_dataloads["arr"]).unsqueeze(0)], dim=0)
                    sam_feature = torch.cat([sam_feature, torch.from_numpy(sam_dataloads["arr"]).unsqueeze(0)], dim=0)
            
        else:
            # print("pairlen: ", pairlen)
            cls_list = []
            imgid_list = []
            for i in range(pairlen):
                clip_feature_path = self.dataset_base + "/clip_features/" + self.pairs[personid][0][i] 
                clip_dataloads = np.load(clip_feature_path)
                clip_cls = torch.from_numpy(clip_dataloads["arr"])[0]
                cls_list.append(clip_cls)
                imgid_list.append(self.pairs[personid][0][i])
            
            vectors = torch.stack(cls_list, dim=0)
            
            normalized_vectors = F.normalize(vectors, p=2, dim=1)

            normalized_vectors = normalized_vectors.numpy()

            kmeans = KMeans(n_clusters=self.limitation, random_state=0)

            cluster_labels = kmeans.fit_predict(normalized_vectors)

            cluster_centers = kmeans.cluster_centers_
            
            cosine_sims = cosine_similarity(normalized_vectors, cluster_centers)

            for i in range(self.limitation):
                cluster_indices = np.where(cluster_labels == i)[0]
                cluster_similarities = cosine_sims[cluster_indices, i]
                representative_index = cluster_indices[np.argmax(cluster_similarities)]
                selected_imgid = imgid_list[representative_index]
                # print("index: ", representative_index)
                
                clip_feature_path = self.dataset_base + "/clip_features/" + selected_imgid 
                sam_feature_path = self.dataset_base + "/sam_features/" + selected_imgid 
                clip_dataloads = np.load(clip_feature_path)
                sam_dataloads = np.load(sam_feature_path)
                if clip_feature is None:
                    clip_feature = torch.from_numpy(clip_dataloads["arr"]).unsqueeze(0)
                    sam_feature = torch.from_numpy(sam_dataloads["arr"]).unsqueeze(0)
                else:
                    clip_feature = torch.cat([clip_feature, torch.from_numpy(clip_dataloads["arr"]).unsqueeze(0)], dim=0)
                    sam_feature = torch.cat([sam_feature, torch.from_numpy(sam_dataloads["arr"]).unsqueeze(0)], dim=0)
         
        return clip_feature, sam_feature, torch.tensor(self.pairs[personid][1])

def build_mlp_dataloader():
# (batchsize, limitation, 677, 1408) (batchsize, limitation, 256, 4096) 6*caption
    batch_size = 6
    shuffle = True
    datas = Dataset_2023us()
    custom_dataloader = DataLoader(datas, batch_size=batch_size, shuffle=shuffle)
    return custom_dataloader 



class MLPRunner(RunnerBase):
    def __init__(
        self,
        model,
        cfg,
    ):
        config = self.build_config(cfg)
        optimizer = self.build_optimizer(model, config)
        dataloader = build_mlp_dataloader()
        max_epoch = config["run"]["max_epoch"]
        device = config["run"]["device"]
        # device = torch.device('cpu')
        super().__init__(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            max_epoch=max_epoch,
            device=device,
        )
        self.config = config
    
    def train_step(self, samples):
        clip_shape = samples[0].shape
        sam_shape = samples[1].shape
        
        # print(samples[1].view(clip_shape[0], clip_shape[1]*clip_shape[2], clip_shape[3]).shape)
        my_samples = {
            'sam_features': samples[1].view(sam_shape[0], sam_shape[1]*sam_shape[2], sam_shape[3]).to(self.device),
            'clip_features': samples[0].view(clip_shape[0], clip_shape[1]*clip_shape[2], clip_shape[3]).to(self.device),
            'target': samples[2].to(self.device),
        }
        # print(my_samples)
        # model = model.to(self.device)
        loss = self.model(my_samples)
        return loss

    def train_epoch(self):
        for samples in tqdm(self.dataloader):
            with torch.cuda.amp.autocast(enabled=True):
                loss = self.train_step(samples)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

    def epoch_start_hook(self, info):
        pass

    def epoch_end_hook(self, info):
        # also save MLP
        # 
        # samblip: Qformer, 32*query token, project(linear)
        # mlpcls: cls token, mlp
        torch.save({
            'epoch': info['cur_epoch'],  # 假设你训练了5个epochs
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, f"/home/xcg/medical-research/Project23us/checkpoints/mlp_checkpoint_{info['cur_epoch']}.pth")
        # }, f"/data2/xcg_data/lavis_data/Breast_images/checkpoint/breast_checkpoint_2nd_{info['cur_epoch']}.pth")
        print(info)

    def build_config(self, cfg):
        with open(cfg, 'r') as file:
            _config = yaml.load(file, Loader=yaml.FullLoader)
        return _config
    
    @classmethod
    def build_optimizer(self, model, config):
        # lr_scale = config["run"]["lr_layer_decay"]
        # weight_decay = config["run"]["weight_decay"]
        optim_params = model.parameters()
        # optim_params = self.model.Parameters()

        # num_parameters = 0
        # for p_group in optim_params:
        #     for p in p_group["params"]:
        #         num_parameters += p.data.nelement()    
        # logging.info("number of trainable parameters: {}".format(num_parameters))      
                
        beta2 = config["run"]["beta2"]

        _optimizer = torch.optim.AdamW(
            optim_params,
            lr=float(config["run"]["init_lr"]),
            betas=(0.9, beta2),
        )    
        return _optimizer
    

model = ClassficationMLP()

device = torch.device("cuda:3")
model.load_state_dict(torch.load("/home/xcg/medical-research/Project23us/checkpoints/mlp_untrained_0.pth", map_location = "cpu"))
model = model.to(device)
cfg = "/home/xcg/medical-research/Project23us/config/train.yaml"
runner = MLPRunner(model, cfg)
runner.train()
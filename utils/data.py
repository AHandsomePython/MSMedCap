# Dataset and Dataloader

import torch
from torch.utils.data import Dataset, DataLoader
import os
import csv
import numpy as np
import yaml

import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# dataset_base = "/data2/xcg_data/lavis_data/2023us/features"
# csvpath = "/data/xcg/lavis_data/coco-2023us/excels/translated.csv"

# Define your custom dataset class
class Dataset_2023us(Dataset):

    def __init__(self, dataset_base, csvpath, limitation = 4):
        self.csvpath = csvpath
        self.dataset_base = dataset_base
        self.pairs, self.keylist = self.load_caption()
        self.limitation = limitation
    
    def load_caption(self):

        # {personid: [[image_ids, ], caption], }
        pairs = {}
        with open(self.csvpath, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                personid = str(row[0]).split("_")[1]
                if personid not in pairs:
                    pairs[personid] = [[row[0]], row[1]]
                else:
                    pairs[personid][0].append(row[0])

        return pairs, list(pairs.keys())


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
                clip_feature_path = self.dataset_base + "/clip_features/" + self.pairs[personid][0][i % pairlen] + '.npz'
                sam_feature_path = self.dataset_base + "/sam_features/" + self.pairs[personid][0][i % pairlen] + '.npz'
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
                clip_feature_path = self.dataset_base + "/clip_features/" + self.pairs[personid][0][i] + '.npz'
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
                
                clip_feature_path = self.dataset_base + "/clip_features/" + selected_imgid + '.npz'
                sam_feature_path = self.dataset_base + "/sam_features/" + selected_imgid + '.npz'
                clip_dataloads = np.load(clip_feature_path)
                sam_dataloads = np.load(sam_feature_path)
                if clip_feature is None:
                    clip_feature = torch.from_numpy(clip_dataloads["arr"]).unsqueeze(0)
                    sam_feature = torch.from_numpy(sam_dataloads["arr"]).unsqueeze(0)
                else:
                    clip_feature = torch.cat([clip_feature, torch.from_numpy(clip_dataloads["arr"]).unsqueeze(0)], dim=0)
                    sam_feature = torch.cat([sam_feature, torch.from_numpy(sam_dataloads["arr"]).unsqueeze(0)], dim=0)
         
        return clip_feature, sam_feature, self.pairs[personid][1]
    
def build_dataloader(cfg):
    # (batchsize, limitation, 677, 1408) (batchsize, limitation, 256, 4096) 6*caption
    batch_size = cfg["dataset"]["batch_size"]
    limitation = cfg["dataset"]["limitation"]
    shuffle = cfg["dataset"]["shuffle"]
    dataset_base = cfg["dataset"]["dataset_base"]
    csvpath = cfg["dataset"]["csv_path"]
    datas = Dataset_2023us(dataset_base=dataset_base, csvpath=csvpath, limitation = limitation)
    custom_dataloader = DataLoader(datas, batch_size=batch_size, shuffle=shuffle)
    return custom_dataloader 


def build_dataloader_from_yaml(config_path):
    with open(config_path, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
     
    return build_dataloader(cfg)
    

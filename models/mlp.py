import torch
import torch.nn as nn
import contextlib
from transformers import AutoTokenizer, OPTForCausalLM, OPTConfig
from models.Qformer import *
import torch.nn.functional as F
import numpy as np
# from transformers import BertTokenizer
# from models.helper.Qformer_helper import BertConfig, BertLMHeadModel

# class MLP_Qformer(nn.Module):
#     def __init__(
#         self,
#         fecture_vec_len=4096,   # 4096 for sam, 1408 for clip
#         num_query_token=32,
#         cross_attention_freq=2,
#         cls=False,
#     ):
#         super().__init__()
#         self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, fecture_vec_len, cross_attention_freq)
#         if cls == False:
#             self.Qformer.cls = None
#             self.Qformer.bert.embeddings.word_embeddings = None
#             self.Qformer.bert.embeddings.position_embeddings = None
#             for layer in self.Qformer.bert.encoder.layer:
#                 layer.output = None
#                 layer.intermediate = None

    
#     def forward(
#         self,
#         features,
#         attention_mask,
#     ):  
#         sam_query_tokens = self.query_tokens.expand(features.shape[0], -1, -1)
#         sam_query_output = self.Qformer.bert(
#             query_embeds=sam_query_tokens,
#             encoder_hidden_states=features,
#             encoder_attention_mask=attention_mask,
#             return_dict=True,
#         )
#         return sam_query_output.last_hidden_state


#     @classmethod # used for stage2 training
#     def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
#         encoder_config = BertConfig.from_pretrained("bert-base-uncased")
#         encoder_config.encoder_width = vision_width
#         encoder_config.add_cross_attention = True
#         encoder_config.cross_attention_freq = cross_attention_freq
#         encoder_config.query_length = num_query_token
#         Qformer = BertLMHeadModel.from_pretrained("bert-base-uncased", config=encoder_config)
#         query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
        
#         query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
#         return Qformer, query_tokens


class ClassficationMLP(nn.Module):
    '''
    'heart', 'liver', 'kidney', 'spleen', 'pancreas', 'breast', 'bladder', 'uterus', 'ovary', 'gallbladder', 'thyroid'
    '''

    def __init__(
            self,
            num_query_token=32,
            clip_vec_len=1408,
            sam_vec_len=4096,
            cls_num = 11, 
            mask = np.ones(11)
        ):
        super().__init__()
        self.clip_qformer = Qformer(fecture_vec_len=clip_vec_len, num_query_token=num_query_token, cross_attention_freq=2, extra_cls=True)
        self.sam_qformer = Qformer(fecture_vec_len=sam_vec_len, num_query_token=num_query_token, cross_attention_freq=2, extra_cls =True)
        self.cls_num = cls_num
        
        self.mlp = nn.Sequential(
            nn.Linear(1536, 4096), nn.ReLU(),
            nn.Linear(4096, 2048), nn.ReLU(),
            nn.Linear(2048, 1024), nn.ReLU(),
            nn.Linear(1024, 128), nn.ReLU(),
            # nn.Linear(1536, 128), nn.ReLU(), for test
        )
        # self.fc1 = nn.Linear(1536, 100)  # 输入大小为1536，输出大小为100
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(100, 42)    # 输入大小为100，输出大小为10
        # self.sigmoid = nn.Sigmoid()
        self.heads = nn.ModuleList([nn.Linear(128, 3) for _ in range(cls_num)])
        # self.lossfn = F.binary_cross_entropy
        # self.lossfn = torch.nn.BCEWithLogitsLoss()
        self.lossfn = torch.nn.CrossEntropyLoss()
        self.mask = mask
        # self.target = torch.rand((4,6*cls_num,))  # will be deleted soon 

    def forward(self, samples):
        clip_features = samples['clip_features']
        sam_features = samples['sam_features']
        target = samples['target']
        target = target[:, 15:18]
        target = torch.argmax(target, dim=1) # for test
        # print("target shape: ", target.shape)
        # print("target: ", target)
        cur_device = clip_features.device
        # print("clip shape: ", clip_features.shape, "sam shape: ", sam_features.shape)

        # CLIP
        clip_attention_mask = torch.ones(clip_features.size()[:-1], dtype=torch.long).to(cur_device)
        # clip_attention_mask[..., -1] = 1
        clip_query_output = self.clip_qformer(
            features=clip_features,
            attention_mask=clip_attention_mask
        )
        
        # SAM
        sam_attention_mask = torch.ones(sam_features.size()[:-1], dtype=torch.long).to(cur_device)
        # sam_attention_mask[..., -1] = 1
        sam_features = sam_features.to(cur_device)
        sam_query_output = self.sam_qformer(
            features=sam_features,
            attention_mask=sam_attention_mask,
        )

        clip_cls = sam_query_output[:, -1, :]
        sam_cls = clip_query_output[:, -1, :] # swap

        cat_cls = torch.cat([clip_cls, sam_cls], dim=1)
        # print(cat_cls.shape)
        # x = self.fc1(cat_cls)
        # x = self.relu(x)
        # x = self.fc2(x)
        x = self.mlp(cat_cls)
        if self.mask.all() == 1:
            x_s = [head(x) for head in self.heads]
            x = torch.stack(x_s, dim=1)
            # x = F.softmax(x, dim=2)
            x = x.view(-1, 3*self.cls_num)
        else:
            # good 1,0,0    bad 0,0,1
            for i in range(self.cls_num):
                x_s = []
                cnt = 0
                if self.mask[i] == 1:
                    cnt+=1
                    x_s.append(self.heads[i](x))
                x = torch.stack(x_s, dim=1)
                # x = F.softmax(x, dim=2)
                x = x.view(-1, 3*cnt)
                    
        # print(x, x.shape, target)
        x = x[:, 15:18]
        # print("x shape: ", x.shape)
        # print("x: ", x)
        loss = self.lossfn(x, target)
        # print("current loss: ", loss.float())
        return loss



    def predict_cls(self, samples):
        clip_features = samples['clip_features']
        sam_features = samples['sam_features']
        # prompt = samples['prompt']
        cur_device = clip_features.device
        with torch.no_grad():
            # CLIP
            clip_attention_mask = torch.zeros(clip_features.size()[:-1], dtype=torch.long).to(cur_device)
            clip_attention_mask[..., -1] = 1
            clip_query_output = self.clip_qformer(
                features=clip_features,
                attention_mask=clip_attention_mask
            )
            
            # SAM
            sam_attention_mask = torch.zeros(sam_features.size()[:-1], dtype=torch.long).to(cur_device)
            sam_attention_mask[..., -1] = 1
            sam_features = sam_features.to(cur_device)
            sam_query_output = self.sam_qformer(
                features=sam_features,
                attention_mask=sam_attention_mask,
            )

            sam_cls = clip_query_output[:, -1, :]
            clip_cls = sam_query_output[:, -1, :] # swap

            cat_cls = torch.cat([clip_cls, sam_cls], dim=1)
            # print(cat_cls.shape)
            # x = self.fc1(cat_cls)
            # x = self.relu(x)
            # x = self.fc2(x)
            x = self.mlp(cat_cls)
            x_s = [head(x) for head in self.heads]
            x = torch.stack(x_s, dim=1)
            x = F.softmax(x, dim=2)
            x = x.view(-1, 3*self.cls_num)
            # print(x)
            return x
            
            
                


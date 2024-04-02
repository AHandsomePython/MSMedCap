import torch
import torch.nn as nn
import contextlib
from transformers import BertTokenizer
from models.helper.Qformer_helper import BertConfig, BertLMHeadModel



class Qformer(nn.Module):
    def __init__(
        self,
        fecture_vec_len=4096,   # 4096 for sam, 1408 for clip
        num_query_token=32,
        cross_attention_freq=2,
        cls=False,
        extra_cls = False
    ):
        super().__init__()
        # lwj revised here
        self.extra_cls = extra_cls
        # lwj revised here
        self.Qformer, self.query_tokens, self.cls_token = self.init_Qformer(num_query_token, fecture_vec_len, cross_attention_freq, extra_cls)
        if cls == False:
            self.Qformer.cls = None
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None

    
    def forward(
        self,
        features,
        attention_mask,
    ):  
        # lwj revised here
        if (self.extra_cls == False):
            sam_query_tokens = self.query_tokens.expand(features.shape[0], -1, -1)
            sam_query_output = self.Qformer.bert(
                query_embeds=sam_query_tokens,
                encoder_hidden_states=features,
                encoder_attention_mask=attention_mask,
                return_dict=True,
            )
        else:
            tokens = torch.cat((self.query_tokens, self.cls_token), dim = 1)
            tokens = tokens.expand(features.shape[0], -1, -1)
            sam_query_output = self.Qformer.bert(
                query_embeds=tokens,
                encoder_hidden_states=features,
                encoder_attention_mask=attention_mask,
                return_dict=True,
            )
        # lwj revised here
        return sam_query_output.last_hidden_state


    @classmethod # used for stage2 training
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2, extra_cls = False):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained("bert-base-uncased", config=encoder_config)
        
        # benlu
        if(extra_cls == False):
            query_tokens = nn.Parameter(torch.randn(1, num_query_token, encoder_config.hidden_size))
            cls_token = nn.Parameter(torch.randn(1, 1, encoder_config.hidden_size))
            cls_token.requires_grad = False #阻断cls
            query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
            cls_token.data.normal_(mean=0.0, std=encoder_config.initializer_range)
            
        if(extra_cls == True):
            query_tokens = nn.Parameter(torch.randn(1, num_query_token, encoder_config.hidden_size))
            cls_token = nn.Parameter(torch.randn(1, 1, encoder_config.hidden_size)) # randn
            query_tokens.requires_grad = False #阻断query token
            # Qformer.requires_grad = False
            for name, param in Qformer.named_parameters():
                param.requires_grad = False
            query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
            cls_token.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        # benlu
            
        return Qformer, query_tokens, cls_token
            
        
        


import torch
import torch.nn as nn
import contextlib
from transformers import AutoTokenizer, OPTForCausalLM, OPTConfig
from models.Qformer import *

class SamBlip(nn.Module):
    def __init__(
            self,
            num_query_token=32,
            opt_model="facebook/opt-2.7b",
            prompt="",
            max_txt_len=320,
            clip_vec_len=1408,
            sam_vec_len=4096
        ):
        super().__init__()
        self.clip_qformer = Qformer(fecture_vec_len=clip_vec_len, num_query_token=num_query_token, cross_attention_freq=2)
        self.sam_qformer = Qformer(fecture_vec_len=sam_vec_len, num_query_token=num_query_token, cross_attention_freq=2)

        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model, torch_dtype=torch.float16
        )
        for _, param in self.opt_model.named_parameters():
            param.requires_grad = False
        self.max_txt_len = max_txt_len
        self.prompt = prompt
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]
    
        self.clip_proj = nn.Linear(
            self.clip_qformer.Qformer.config.hidden_size,
            self.opt_model.config.hidden_size
        )
        self.sam_proj = nn.Linear(
            self.sam_qformer.Qformer.config.hidden_size,
            self.opt_model.config.hidden_size
        )


    def forward(self, samples):
        clip_features = samples['clip_features']
        sam_features = samples['sam_features']
        text_input = samples['text_input']
        cur_device = clip_features.device
        
        # CLIP
        clip_query_output = self.clip_qformer(
            features=clip_features,
            attention_mask=torch.ones(clip_features.size()[:-1], dtype=torch.long).to(cur_device)
        )
        clip_inputs_opt = self.clip_proj(clip_query_output)
        clip_atts_opt = torch.ones(clip_inputs_opt.size()[:-1], dtype=torch.long).to(
            cur_device
        )

        # SAM
        sam_features = sam_features.to(cur_device)
        sam_query_output = self.sam_qformer(
            features=sam_features,
            attention_mask=torch.ones(sam_features.size()[:-1], dtype=torch.long).to(cur_device),
        )
        sam_inputs_opt = self.sam_proj(sam_query_output)
        sam_atts_opt = torch.ones(sam_inputs_opt.size()[:-1], dtype=torch.long).to(
            cur_device
        )


        # LLM
        self.opt_tokenizer.padding_side = "right"
        text = [t + "\n" for t in text_input]
        opt_tokens = self.opt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(cur_device)
        targets = opt_tokens.input_ids.masked_fill(
            opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt
        empty_targets = (
            torch.ones(clip_atts_opt.size(), dtype=torch.long).to(cur_device).fill_(-100)
        )
        sam_empty_targets = (
            torch.ones(sam_atts_opt.size(), dtype=torch.long).to(cur_device).fill_(-100)
        )
        targets = torch.cat([empty_targets, sam_empty_targets, targets], dim=1)
        inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        inputs_embeds = torch.cat([clip_inputs_opt, sam_inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([clip_atts_opt, sam_atts_opt, opt_tokens.attention_mask], dim=1)
        with self.maybe_autocast():
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss
        return {"loss": loss}


    @torch.no_grad()
    def generate(
        self,
        samples,
        # clip_features,
        # sam_features,
        # prompt,
        use_nucleus_sampling=False,
        num_beams=15,
        max_length=50,
        min_length=10,
        top_p=0.9,
        repetition_penalty=9.0,
        length_penalty=1.2,
        num_captions=1,
        temperature=0.5,
    ):
        clip_features = samples['clip_features']
        sam_features = samples['sam_features']
        prompt = samples['prompt']
        cur_device = clip_features.device
        with self.maybe_autocast():
            # CLIP
            clip_query_output = self.clip_qformer(
                features=clip_features,
                attention_mask=torch.ones(clip_features.size()[:-1], dtype=torch.long).to(cur_device)
            )
            clip_inputs_opt = self.clip_proj(clip_query_output)
            clip_atts_opt = torch.ones(clip_inputs_opt.size()[:-1], dtype=torch.long).to(
                cur_device
            )

            # SAM
            sam_features = sam_features.to(cur_device)
            sam_query_output = self.sam_qformer(
                features=sam_features,
                attention_mask=torch.ones(sam_features.size()[:-1], dtype=torch.long).to(cur_device),
            )
            sam_inputs_opt = self.sam_proj(sam_query_output)
            sam_atts_opt = torch.ones(sam_inputs_opt.size()[:-1], dtype=torch.long).to(
                cur_device
            )

            # LLM
            opt_tokens = self.opt_tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(cur_device)
            attention_mask = torch.cat([clip_atts_opt, sam_atts_opt, opt_tokens.attention_mask], dim=1)
            prompt_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([clip_inputs_opt, sam_inputs_opt, prompt_embeds], dim=1)
            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds, 
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
            return output_text
    

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        # enable_autocast = self.device != torch.device("cpu")
        enable_autocast = True

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def get_optimizer_params(self, weight_decay, lr_scale=1):

        # vit_num_layers = self.visual_encoder.get_num_layer()
        # lr_scales = list(lr_scale ** (vit_num_layers + 1 - i) for i in range(vit_num_layers + 2))

        parameter_group_names = {}
        parameter_group_vars = {}

        for name, param in self.named_parameters():
            # print(name,end=None)
            if not param.requires_grad:
                # print(" continue==========")
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias"):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay
            if 'visual_encoder' in name:
                layer_id = self.visual_encoder.get_num_layer(name.replace('visual_encoder.',''))
                group_name = "vit_layer_%d_%s" % (layer_id, group_name)
            else:
                layer_id = None

            if group_name not in parameter_group_names:
                if layer_id is not None:
                    # scale = lr_scales[layer_id]
                    print("layer_id is not None: ")
                    pass
                else:
                    scale = 1
                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)
        # import json
        # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
        optim_params = list(parameter_group_vars.values())
        return optim_params